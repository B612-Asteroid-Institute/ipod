import logging
from typing import Optional, Tuple, Type, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from adam_core.coordinates.residuals import calculate_reduced_chi2
from adam_core.orbit_determination import (
    OrbitDeterminationObservations,
    evaluate_orbits,
)
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.propagator.adam_pyoorb import PYOORBPropagator as PYOORB
from precovery.precovery_db import PrecoveryCandidatesQv as PrecoveryCandidates
from precovery.precovery_db import PrecoveryDatabase
from thor.orbit_determination import FittedOrbitMembers, FittedOrbits
from thor.orbits.iod import iod
from thor.orbits.od import od

from .utils import (
    check_candidates_astrometric_errors,
    compute_search_window_and_tolerance,
    drop_coincident_candidates,
    identify_found_missed_and_new,
)

logger = logging.getLogger(__name__)


class SearchSummary(qv.Table):
    orbit_id = qv.LargeStringColumn()
    min_mjd = qv.Float64Column()
    max_mjd = qv.Float64Column()
    num_candidates = qv.Int64Column()
    num_accepted = qv.Int64Column()
    num_rejected = qv.Int64Column()
    arc_length_prev = qv.Float64Column()
    arc_length = qv.Float64Column()
    num_obs_prev = qv.Int64Column()
    num_obs = qv.Int64Column()


class OrbitOutliers(qv.Table):
    orbit_id = qv.LargeStringColumn(nullable=True)
    obs_id = qv.LargeStringColumn()

    def select_global_outliers(self):
        return self.apply_mask(pc.is_null(self.orbit_id))

    def select_orbit_outliers(self, orbit_id):
        return self.select("orbit_id", orbit_id)

    def select_orbit_and_global_outliers(self, orbit_id):
        null_mask = pc.is_null(self.orbit_id)
        orbit_mask = pc.equal(self.orbit_id, orbit_id)
        orbit_mask = pc.fill_null(orbit_mask, False)
        return self.apply_mask(pc.or_(null_mask, orbit_mask))


DEFAULT_ASTROMETRIC_ERRORS = {
    # Default errors in arcseconds (100 mas, 100 mas)
    "default": (0.1, 0.1),
}


def ipod(
    orbit: Union[Orbits, FittedOrbits],
    orbit_observations: Optional[OrbitDeterminationObservations] = None,
    min_tolerance: float = 1.0,
    max_tolerance: float = 10.0,
    tolerance_step: float = 5.0,
    delta_time: float = 15.0,
    rchi2_threshold: float = 3.0,
    outlier_chi2: float = 9.0,
    reconsider_chi2: float = 8.0,
    min_mjd: Optional[float] = None,
    max_mjd: Optional[float] = None,
    astrometric_errors: Optional[dict[str, Tuple[float, float]]] = None,
    database: Union[str, PrecoveryDatabase] = "",
    datasets: Optional[set[str]] = None,
    orbit_outliers: Optional[OrbitOutliers] = None,
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
) -> Tuple[FittedOrbits, FittedOrbitMembers, PrecoveryCandidates, SearchSummary]:
    logger.debug(f"Running ipod with orbit {orbit.orbit_id[0].as_py()}...")
    if astrometric_errors is None:
        astrometric_errors = DEFAULT_ASTROMETRIC_ERRORS

    # Initialize the propagator
    prop = propagator(**propagator_kwargs)

    # If we haven't received an active precovery database then
    # create a connection to one
    if not isinstance(database, PrecoveryDatabase):
        precovery_db = PrecoveryDatabase.from_dir(
            database,
            create=False,
            mode="r",
            allow_version_mismatch=True,
        )
    else:
        precovery_db = database

    # Get the mjd bounds of the frames in the database
    database_min_mjd, database_max_mjd = precovery_db.frames.idx.mjd_bounds(
        datasets=datasets
    )
    logger.debug(f"Database MJD range: {database_min_mjd:.5f} - {database_max_mjd:.5f}")
    if database_min_mjd is None and database_max_mjd is None:
        if datasets is not None:
            logger.debug(f"No frames for datasets {datasets} in the database.")
        else:
            logger.debug("No frames in the database.")
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            PrecoveryCandidates.empty(),
            SearchSummary.empty(),
        )

    # If the min and max MJD are not given then set them to the min and max MJD
    # of the frames in the database
    if min_mjd is None:
        min_mjd = database_min_mjd
    if max_mjd is None:
        max_mjd = database_max_mjd

    # If the min and max MJD are outside the bounds of the frames in the database
    # then set them to the min and max MJD of the frames in the database
    if min_mjd < database_min_mjd:
        min_mjd = database_min_mjd
        logger.debug(
            "Minimum MJD is before the earliest frame in the database. Setting to the earliest frame."
        )
    if max_mjd > database_max_mjd:
        max_mjd = database_max_mjd
        logger.debug(
            "Maximum MJD is after the latest frame in the database. Setting to the latest frame."
        )

    force_fit = False
    orbit_observations_iter = OrbitDeterminationObservations.empty()
    # If observations have been passed lets make sure that
    # the given orbit has been evaluated with the given observations
    if orbit_observations is not None:
        # Make sure the observations are sorted
        orbit_observations = orbit_observations.sort_by(
            [
                "coordinates.time.days",
                "coordinates.time.nanos",
                "coordinates.origin.code",
            ]
        )

        # mask out permanent rejections
        # should we be doing this this early?
        n_obs = len(orbit_observations)
        if orbit_outliers is not None:
            orbit_observations = orbit_observations.apply_mask(
                pc.invert(
                    pc.is_in(
                        orbit_observations.id,
                        orbit_outliers.select_orbit_and_global_outliers(
                            orbit.orbit_id[0]
                        ).obs_id,
                    )
                )
            )
            logger.debug(
                f"Removed {n_obs - len(orbit_observations)} global"
                " and orbit-specific rejections."
            )

        # Evaluate the orbit with the given observations if we have
        # any left
        if len(orbit_observations) > 0:
            orbit_iter, orbit_members_iter = evaluate_orbits(
                orbit.to_orbits(),
                orbit_observations,
                prop,
            )

            # We recast ouput orbits and orbit members to the FittedOrbits
            # and FittedOrbitMember from THOR
            orbit_iter = FittedOrbits.from_kwargs(
                orbit_id=orbit_iter.orbit_id,
                coordinates=orbit_iter.coordinates,
                arc_length=orbit_iter.arc_length,
                num_obs=orbit_iter.num_obs,
                chi2=orbit_iter.chi2,
                reduced_chi2=orbit_iter.reduced_chi2,
                iterations=orbit_iter.iterations,
                success=orbit_iter.success,
                status_code=orbit_iter.status_code,
            )
            orbit_members_iter = FittedOrbitMembers.from_kwargs(
                orbit_id=orbit_members_iter.orbit_id,
                obs_id=orbit_members_iter.obs_id,
                residuals=orbit_members_iter.residuals,
                outlier=orbit_members_iter.outlier,
                solution=orbit_members_iter.solution,
            )

        else:
            logger.debug(
                "No observations left after removing global and orbit-specific rejections. "
                "Proceeding with previous orbit and ignoring provided observations."
            )
            orbit_iter = orbit
            orbit_members_iter = FittedOrbitMembers.empty()
            orbit_observations_iter = OrbitDeterminationObservations.empty()
            obs_ids_iter = pa.array([])

        # Now fit the orbit with the given observations so we make sure we have
        # an accurate orbit to start with

        # Run IOD only if we have been given a set of outliers and we've
        # removed observations from the orbit_observations
        if n_obs != len(orbit_observations) and len(orbit_observations) > 0:
            orbit_iter_fit, orbit_members_iter_fit = iod(
                orbit_observations,
                min_obs=np.minimum(len(orbit_observations), 6),
                min_arc_length=1.0,
                contamination_percentage=0.0,
                rchi2_threshold=1e5,
                observation_selection_method="combinations",
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
            )
            if len(orbit_iter_fit) > 0:
                orbit_iter_fit = orbit_iter_fit.set_column(
                    "orbit_id", orbit_iter.orbit_id
                )
                orbit_members_iter_fit = orbit_members_iter_fit.set_column(
                    "orbit_id",
                    pa.repeat(orbit_iter.orbit_id[0], len(orbit_members_iter_fit)),
                )
        else:
            orbit_iter_fit = FittedOrbits.empty()
            orbit_members_iter_fit = FittedOrbitMembers.empty()

        # What if IOD identifies outliers???
        if len(orbit_iter_fit) != 0:
            orbit_iter_fit, orbit_members_iter_fit = od(
                orbit_iter_fit,
                orbit_observations,
                rchi2_threshold=rchi2_threshold,
                min_obs=6,
                min_arc_length=1.0,
                contamination_percentage=0.0,
                delta=1e-8,
                max_iter=5,
                method="central",
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
            )

        if len(orbit_iter_fit) == 0:
            logger.debug(
                "No valid observations found. Proceeding with previous orbit and "
                "ignoring provided observations."
            )
            orbit_iter = orbit
            orbit_members_iter = FittedOrbitMembers.empty()
            orbit_observations_iter = OrbitDeterminationObservations.empty()
            obs_ids_iter = pa.array([])

    else:
        orbit_iter = orbit
        orbit_members_iter = FittedOrbitMembers.empty()
        orbit_observations_iter = OrbitDeterminationObservations.empty()
        obs_ids_iter = pa.array([])

    # Start with a tolerance of 1 arcsecond
    tolerances = np.arange(
        min_tolerance, max_tolerance + tolerance_step, tolerance_step
    )
    tolerance_iter = tolerances[0]
    logger.debug(f"Attempting ipod with {len(tolerances)} tolerances.")

    # Running list of observation IDs that have been rejected by OD
    # or OD failures
    rejected_obs_ids: set[str] = set()
    # Track the observation IDs that have been processed. This list
    # will only ever grow as more observations are found and tested
    # against the orbit
    processed_obs_ids: set[str] = set()
    candidates_iter = PrecoveryCandidates.empty()

    # Running search summary
    search_summary_iter = {
        "orbit_id": orbit_iter.orbit_id[0].as_py(),
        "min_mjd": 0,
        "max_mjd": 0,
        "num_candidates": 0,
        "num_accepted": 0,
        "num_rejected": 0,
        "arc_length_prev": orbit_iter.arc_length[0].as_py(),
        "arc_length": 0,
        "num_obs_prev": orbit_iter.num_obs[0].as_py(),
        "num_obs": 0,
    }

    failed_corrections = 0
    i = 0
    while tolerance_iter < max_tolerance:
        i += 1
        logger.debug(f"Starting IPOD iteration {i}...")

        if len(orbit_observations_iter) > 0:
            # Update the running list of processed observation IDs
            processed_obs_ids.update(obs_ids_iter.to_pylist())

        # Compute the min and max observation times if they exist
        if len(orbit_observations_iter) > 0:
            min_search_mjd = (
                orbit_observations_iter.coordinates.time.min().mjd()[0].as_py()
            )
            max_search_mjd = (
                orbit_observations_iter.coordinates.time.max().mjd()[0].as_py()
            )

        # If they do not exist then set the min and max search to epoch of the orbit
        else:
            min_search_mjd = max_search_mjd = (
                orbit_iter.coordinates.time.rescale("utc").mjd()[0].as_py()
            )

        # Compute the "optimal" search window and tolerance
        min_mjd_iter, max_mjd_iter, _ = compute_search_window_and_tolerance(
            orbit_iter.to_orbits(),
            prop,
            max_tolerance=tolerance_iter,
        )

        # If the "optimal" search window doesn't span the entire observation time range of this
        # iteration then set the search window to the entire observation time range
        if not np.isnan(min_mjd_iter) and not np.isnan(max_mjd_iter):
            if min_mjd_iter > min_search_mjd:
                min_mjd_iter = min_search_mjd
                logger.debug(
                    "Proposed search window start is after the earliest observation MJD."
                )
            if max_mjd_iter < max_search_mjd:
                max_mjd_iter = max_search_mjd
                logger.debug(
                    "Proposed search window end is before the latest observation MJD."
                )

        # If the search window time range is nan which can occur for highly uncertain
        # orbits then set the tolerance to the starting tolerance and set the
        # maximum search window to the entire observation time range
        if np.isnan(min_mjd_iter) or np.isnan(max_mjd_iter):
            logger.debug(
                "Orbit's on-sky uncertainty already exceeds the maximum tolerance. "
                f"Setting search window to +-{delta_time} and tolerance to {tolerance_iter:.3f}."
            )
            min_mjd_iter = min_search_mjd - delta_time
            max_mjd_iter = max_search_mjd + delta_time

            # If we are setting the search window with a default tolerance then we should
            # force a fit with the given observations (if no new observations are found
            # in this iteration) to give this orbit a chance to improve. We will limit it
            # to 3 failed corrections before forcing a fit.
            if failed_corrections < 3:
                force_fit = True
            else:
                force_fit = False

        # If the search window is outside the minimum and maximum MJD, adjust the search window
        # to be within the minimum and maximum MJD
        if min_mjd is not None and min_mjd_iter is not None:
            min_mjd_iter = np.maximum(min_mjd_iter, min_mjd)
            logger.debug("Proposed search window start is before the minimum MJD.")
        if max_mjd is not None and max_mjd_iter is not None:
            max_mjd_iter = np.minimum(max_mjd_iter, max_mjd)
            logger.debug("Proposed search window end is after the maximum MJD.")

        logger.debug(
            f"Running precovery search for {orbit_iter.orbit_id[0].as_py()} "
            f"between {min_mjd_iter:.5f} and {max_mjd_iter:.5f} "
            f"[dt: {max_mjd_iter-min_mjd_iter:.5f}] with a {tolerance_iter:.3f} arcsecond tolerance..."
        )
        candidates_iter, frame_candidates = precovery_db.precover(
            orbit_iter.to_orbits(),
            tolerance=tolerance_iter / 3600,
            start_mjd=min_mjd_iter,
            end_mjd=max_mjd_iter,
            window_size=1,
            datasets=datasets,
        )
        candidates_iter = candidates_iter.sort_by(
            ["time.days", "time.nanos", "obscode"]
        )
        num_candidates = len(candidates_iter)
        logger.debug(f"Found {num_candidates} potential precovery observations.")

        # Check if the candidates contain any missing uncertainties. If they do then
        # assign a default uncertainty to them.
        candidates_iter = check_candidates_astrometric_errors(
            candidates_iter, astrometric_errors
        )

        # Drop any coincident candidates
        candidates_iter = drop_coincident_candidates(candidates_iter)
        candidates_iter = candidates_iter.sort_by(
            ["time.days", "time.nanos", "obscode"]
        )

        if len(candidates_iter) != num_candidates:
            logger.debug(
                f"Removed {num_candidates - len(candidates_iter)} coincident candidates."
            )
            num_candidates = len(candidates_iter)

        # drop candidates that are in the orbit outliers table
        if orbit_outliers is not None:
            candidates_iter = candidates_iter.apply_mask(
                pc.invert(
                    pc.is_in(
                        candidates_iter.observation_id,
                        orbit_outliers.select_orbit_and_global_outliers(
                            orbit.orbit_id[0]
                        ).obs_id,
                    )
                )
            )

        if len(candidates_iter) != num_candidates:
            logger.debug(
                f"Removed {num_candidates - len(candidates_iter)} observations "
                "that were in the orbit outliers table."
            )
            num_candidates = len(candidates_iter)

        # This only occurs if we were given input observations
        if i == 0 and len(processed_obs_ids) > 0:
            # Check if the candidates contain any observations that were previously
            # given as input observations. If they do then update the search summary
            # number of accepted observations
            found_obs_id_prev, _, _ = identify_found_missed_and_new(
                pa.array(list(processed_obs_ids)), candidates_iter.observation_id
            )
            search_summary_iter["num_accepted"] = len(found_obs_id_prev)
            search_summary_iter["arc_length"] = (
                candidates_iter.time.max().mjd()[0].as_py()
                - candidates_iter.time.min().mjd()[0].as_py()
            )
            search_summary_iter["num_obs"] = len(found_obs_id_prev)

        # Current set of observation IDs up for consideration
        obs_ids_iter = candidates_iter.observation_id

        # Update the search summary
        search_summary_iter["min_mjd"] = min_mjd_iter
        search_summary_iter["max_mjd"] = max_mjd_iter
        search_summary_iter["num_candidates"] = len(processed_obs_ids)

        if len(candidates_iter) < 6:
            logger.debug(
                "Insufficient candidates for orbit determination. "
                f"Increasing tolerance to {tolerance_iter + tolerance_step}."
            )
            tolerance_iter += tolerance_step
            continue

        # Convert candidates to OrbitDeterminationObservations
        orbit_observations_iter = OrbitDeterminationObservations.from_kwargs(
            id=obs_ids_iter,
            coordinates=candidates_iter.to_spherical_coordinates(),
            observers=candidates_iter.get_observers(),
        )

        # The orbit solution may have changed so if we have any rejected observations
        # lets evaluate the ensemble and see if any should be re-accepted
        obs_ids_reconsider: set[str] = set()
        if len(rejected_obs_ids) > 0:
            _, orbit_members_iter_ = evaluate_orbits(
                orbit_iter.to_orbits(),
                orbit_observations_iter,
                prop,
            )

            # We recast ouput orbit members to the FittedOrbits
            orbit_members_iter = FittedOrbitMembers.from_kwargs(
                orbit_id=orbit_members_iter.orbit_id,
                obs_id=orbit_members_iter.obs_id,
                residuals=orbit_members_iter.residuals,
                outlier=orbit_members_iter.outlier,
                solution=orbit_members_iter.solution,
            )

            # Find any observations that have a chi2 less than the reconsider_chi2
            # and add them for re-consideration
            obs_ids_reconsider = set(
                orbit_members_iter_.apply_mask(
                    pc.and_(
                        pc.less_equal(
                            orbit_members_iter_.residuals.chi2, reconsider_chi2
                        ),
                        pc.is_in(
                            orbit_members_iter_.obs_id, pa.array(rejected_obs_ids)
                        ),
                    )
                ).obs_id.to_pylist()
            )
            if len(obs_ids_reconsider) > 0:
                logger.debug(
                    f"Re-accepting {len(obs_ids_reconsider)} previously rejected observations."
                )
                rejected_obs_ids -= obs_ids_reconsider
                search_summary_iter["num_rejected"] = len(rejected_obs_ids)
                force_fit = True

        found_obs_ids, missed_obs_ids, new_obs_ids = identify_found_missed_and_new(
            pa.array(list(processed_obs_ids)), obs_ids_iter
        )
        logger.debug(
            f"Found {len(found_obs_ids)} observations from the previous iteration."
        )
        logger.debug(
            f"Missed {len(missed_obs_ids)} observations from the previous iteration."
        )
        logger.debug(f"Found {len(new_obs_ids)} new observations.")

        # If no new observations were found, then lets increase the tolerance and jump to the next
        # iteration. There is no point in orbit fitting if no new observations were found.
        if len(new_obs_ids) == 0 and not force_fit:
            logger.debug(
                "No new observations found. "
                f"Increasing the tolerance to {tolerance_iter + tolerance_step} arcseconds."
            )
            tolerance_iter += tolerance_step
            continue

        # Remove any observations that we have previously rejected
        if len(rejected_obs_ids) > 0:
            num_candidates = len(orbit_observations_iter)
            orbit_observations_iter = orbit_observations_iter.apply_mask(
                pc.invert(
                    pc.is_in(orbit_observations_iter.id, pa.array(rejected_obs_ids))
                )
            )
            logger.debug(
                f"Removed {num_candidates - len(orbit_observations_iter)} rejected observations."
            )

        # Attempt to differentially correct the orbit given the new observations
        # If we have previous observations then we will use them to calculate
        # the contamination percentage
        if len(orbit_observations_iter) > 0:
            contamination_percentage = np.minimum(
                (len(new_obs_ids) + 1) / len(orbit_observations_iter) * 100, 100
            )

        # If we do not have previous observations then we will use a default
        # maximum contamination percentage of 50%
        else:
            logger.debug(
                "No previous observations. Running orbit fit with all new observations."
            )
            contamination_percentage = 50.0

        logger.debug(
            f"Running orbit fit with {len(new_obs_ids)} new observations "
            f"and a contamination percentage of {contamination_percentage:.2f}%..."
        )
        orbit_iter_fit, orbit_members_iter_fit = od(
            orbit_iter,
            orbit_observations_iter,
            rchi2_threshold=rchi2_threshold,
            min_obs=6,
            min_arc_length=1.0,
            contamination_percentage=contamination_percentage,
            delta=1e-8,
            max_iter=10,
            method="central",
            propagator=propagator,
            propagator_kwargs=propagator_kwargs,
        )
        if force_fit:
            force_fit = False

        if len(orbit_iter_fit) == 0:
            # If the orbit fit failed completely, then remove the newly added observations
            # and add them to the rejected observations list. We are trusting that the OD
            # code did its best job at removing the outliers but that it still failed
            # despite its best efforts. The rejected observations can still be re-accepted
            # in a later iteration if their chi2 is less than the reconsider_chi2.
            failed_corrections += 1
            logger.debug(
                "Orbit fit failed. Removing newly added observations and adding "
                "them to the rejected observations list."
            )
            if len(new_obs_ids) > 0:
                rejected_obs_ids.update(new_obs_ids)

            # If the fit failed and we have added observations for reconsideration
            # then we should re-add them to the rejected list
            if len(obs_ids_reconsider) > 0:
                rejected_obs_ids.update(obs_ids_reconsider)

            # Update the search summary and skip to the next iteration since the
            # best-fit orbit has not converged
            search_summary_iter["num_rejected"] = len(rejected_obs_ids)

            if failed_corrections > 2:
                # If we have failed 3 times then we should increase the tolerance
                # and try again
                logger.debug(
                    "Orbit fit has failed 3 consecutive times. "
                    f"Increasing tolerance to {tolerance_iter + tolerance_step}."
                )
                tolerance_iter += tolerance_step
            continue
        else:
            failed_corrections = 0
            orbit_iter = orbit_iter_fit
            orbit_members_iter = orbit_members_iter_fit

        # Remove any potential outliers: observations already identified
        # by the OD code as outliers or any observations with a chi2
        # greater than the outlier_chi2
        outlier_mask = pc.or_(
            pc.equal(orbit_members_iter.outlier, True),
            pc.greater(orbit_members_iter.residuals.chi2, outlier_chi2),
        )
        outlier_ids = orbit_members_iter.apply_mask(outlier_mask).obs_id

        # Add identified outliers to the rejected observations list
        rejected_obs_ids.update(outlier_ids.to_pylist())

        # Remove the outliers from the orbit observations and orbit members
        orbit_observations_iter = orbit_observations_iter.apply_mask(
            pc.invert(pc.is_in(orbit_observations_iter.id, outlier_ids))
        )
        orbit_members_iter = orbit_members_iter.apply_mask(
            pc.invert(pc.is_in(orbit_members_iter.obs_id, outlier_ids))
        )

        # Get the array of observations that were accepted
        obs_ids_accepted = orbit_observations_iter.id
        logger.debug(
            f"Accepted {len(obs_ids_accepted)} observations and identified {len(outlier_ids)} outliers."
        )

        # Update candidates iter to reflect the new set of observations
        candidates_iter = candidates_iter.apply_mask(
            pc.is_in(candidates_iter.observation_id, orbit_observations_iter.id)
        )

        # Compute the arc length, number of observations, and the reduced chi2
        # of the orbit fit
        arc_length = (
            orbit_observations_iter.coordinates.time.max().mjd()[0].as_py()
            - orbit_observations_iter.coordinates.time.min().mjd()[0].as_py()
        )
        num_obs = len(orbit_observations_iter)
        reduced_chi2 = calculate_reduced_chi2(orbit_members_iter.residuals, 6)

        # Update the rest of the search summary
        search_summary_iter["num_accepted"] = len(obs_ids_accepted)
        search_summary_iter["num_rejected"] = len(rejected_obs_ids)
        search_summary_iter["arc_length"] = arc_length
        search_summary_iter["num_obs"] = len(orbit_observations_iter)

        # Update the orbit's fit parameters since we may have removed some observations
        # not removed by OD
        orbit_iter = orbit_iter.set_column("num_obs", pa.array([num_obs]))
        orbit_iter = orbit_iter.set_column("arc_length", pa.array([arc_length]))
        orbit_iter = orbit_iter.set_column(
            "chi2", pa.array([pc.sum(orbit_members_iter.residuals.chi2)])
        )
        orbit_iter = orbit_iter.set_column("reduced_chi2", pa.array([reduced_chi2]))

        if len(new_obs_ids) == 0:
            # If the observations have not changed since the previous iteration, lets
            # try increasing the search window and tolerance
            if (
                min_mjd == min_mjd_iter
                and max_mjd == max_mjd_iter
                and tolerance_iter >= max_tolerance
            ):
                logger.debug(
                    "Observations have not changed since the previous iteration "
                    "and the search window and tolerance have not changed. Exiting."
                )

                search_summary = SearchSummary.from_kwargs(
                    **{k: [v] for k, v in search_summary_iter.items()}
                )
                return (
                    orbit_iter,
                    orbit_members_iter,
                    candidates_iter,
                    search_summary,
                )

            else:
                logger.debug(
                    "Observations have not changed since the previous iteration. "
                    f"Increasing the tolerance to {tolerance_iter + tolerance_step} arcseconds."
                )
                tolerance_iter += tolerance_step
                continue

    if len(orbit_members_iter) == 0:
        logger.debug("No valid observation found. Exiting.")
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            PrecoveryCandidates.empty(),
            SearchSummary.empty(),
        )

    if orbit_iter.reduced_chi2[0].as_py() > rchi2_threshold:
        logger.debug(
            f"Final reduced chi2 of {orbit_iter.reduced_chi2[0].as_py()} "
            f"is greater than the threshold of {rchi2_threshold}."
        )
        return (
            FittedOrbits.empty(),
            FittedOrbitMembers.empty(),
            PrecoveryCandidates.empty(),
            SearchSummary.empty(),
        )

    # Filter the candidates to only include observations that were used in the final orbit fit
    candidates = candidates_iter.apply_mask(
        pc.is_in(candidates_iter.observation_id, orbit_members_iter.obs_id)
    )

    search_summary = SearchSummary.from_kwargs(
        **{k: [v] for k, v in search_summary_iter.items()}
    )
    # return orbit_outliers ()
    return orbit_iter, orbit_members_iter, candidates, search_summary
