import logging
import multiprocessing as mp
import time
from typing import Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc
import quivr as qv
import ray
from adam_core.orbit_determination import OrbitDeterminationObservations
from adam_core.orbits import Orbits
from adam_core.propagator import Propagator
from adam_core.propagator.adam_pyoorb import PYOORBPropagator as PYOORB
from adam_core.propagator.utils import _iterate_chunk_indices, _iterate_chunks
from adam_core.ray_cluster import initialize_use_ray
from precovery.precovery_db import PrecoveryDatabase
from thor.observations import Observations
from thor.orbit_determination import FittedOrbitMembers, FittedOrbits

from .ipod import OrbitOutliers, PrecoveryCandidates, SearchSummary, ipod

logger = logging.getLogger(__name__)


def ipod_worker(
    orbit_ids: npt.NDArray[np.str_],
    orbits: FittedOrbits,
    orbit_members: Optional[FittedOrbitMembers] = None,
    observations: Optional[Observations] = None,
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

    ipod_orbits = FittedOrbits.empty()
    ipod_orbits_members = FittedOrbitMembers.empty()
    ipod_precovery_candidates = PrecoveryCandidates.empty()
    ipod_summary = SearchSummary.empty()

    if not isinstance(database, PrecoveryDatabase):
        precovery_db = PrecoveryDatabase.from_dir(
            database,
            create=False,
            mode="r",
            allow_version_mismatch=True,
        )
    else:
        precovery_db = database

    for orbit_id in orbit_ids:

        orbit = orbits.select("orbit_id", orbit_id)

        if orbit_members is not None and observations is not None:
            obs_ids = orbit_members.apply_mask(
                pc.equal(orbit_members.orbit_id, orbit_id)
            ).obs_id
            orbit_observations = observations.apply_mask(
                pc.is_in(observations.id, obs_ids)
            )

            # Sort the observations by time and origin
            orbit_observations = orbit_observations.sort_by(
                [
                    "coordinates.time.days",
                    "coordinates.time.nanos",
                    "coordinates.origin.code",
                ]
            )

            # Calculate observers
            observers = orbit_observations.get_observers().observers
            observers = observers.sort_by(
                ["coordinates.time.days", "coordinates.time.nanos", "code"]
            )

            # Create orbit determination observations
            orbit_observations = OrbitDeterminationObservations.from_kwargs(
                id=orbit_observations.id,
                coordinates=orbit_observations.coordinates,
                observers=observers,
            )
        else:
            orbit_observations = None

        try:
            ipod_orbit, ipod_orbit_members, ipod_candidates_i, summary = ipod(
                orbit,
                orbit_observations=orbit_observations,
                min_tolerance=min_tolerance,
                max_tolerance=max_tolerance,
                tolerance_step=tolerance_step,
                delta_time=delta_time,
                rchi2_threshold=rchi2_threshold,
                outlier_chi2=outlier_chi2,
                reconsider_chi2=reconsider_chi2,
                min_mjd=min_mjd,
                max_mjd=max_mjd,
                astrometric_errors=astrometric_errors,
                database=precovery_db,
                datasets=datasets,
                orbit_outliers=orbit_outliers,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
            )
        except Exception as e:
            logger.error(f"Error processing orbit {orbit_id}: {e}")
            print(f"Error processing orbit {orbit_id}: {e}")
            raise e

        ipod_orbits = qv.concatenate([ipod_orbits, ipod_orbit])
        if ipod_orbits.fragmented():
            ipod_orbits = qv.defragment(ipod_orbits)
        ipod_orbits_members = qv.concatenate([ipod_orbits_members, ipod_orbit_members])
        if ipod_orbits_members.fragmented():
            ipod_orbits_members = qv.defragment(ipod_orbits_members)
        ipod_precovery_candidates = qv.concatenate(
            [ipod_precovery_candidates, ipod_candidates_i]
        )
        if ipod_precovery_candidates.fragmented():
            ipod_precovery_candidates = qv.defragment(ipod_precovery_candidates)
        ipod_summary = qv.concatenate([ipod_summary, summary])
        if ipod_summary.fragmented():
            ipod_summary = qv.defragment(ipod_summary)

    return ipod_orbits, ipod_orbits_members, ipod_precovery_candidates, ipod_summary


@ray.remote
def ipod_worker_remote(
    orbit_ids: npt.NDArray[np.str_],
    orbit_ids_indices: Tuple[int, int],
    orbits: Union[Orbits, FittedOrbits],
    orbit_members: Optional[FittedOrbitMembers] = None,
    observations: Optional[Observations] = None,
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
    database_directory: str = "",
    datasets: Optional[set[str]] = None,
    orbit_outliers: Optional[OrbitOutliers] = None,
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
):

    database = PrecoveryDatabase.from_dir(
        database_directory,
        create=False,
        mode="r",
        allow_version_mismatch=True,
    )

    orbit_id_chunk = orbit_ids[orbit_ids_indices[0] : orbit_ids_indices[1]]
    (
        ipod_orbits,
        ipod_orbits_members,
        ipod_precovery_candidates,
        ipod_summary,
    ) = ipod_worker(
        orbit_id_chunk,
        orbits,
        orbit_members=orbit_members,
        observations=observations,
        min_tolerance=min_tolerance,
        max_tolerance=max_tolerance,
        tolerance_step=tolerance_step,
        delta_time=delta_time,
        rchi2_threshold=rchi2_threshold,
        outlier_chi2=outlier_chi2,
        reconsider_chi2=reconsider_chi2,
        min_mjd=min_mjd,
        max_mjd=max_mjd,
        astrometric_errors=astrometric_errors,
        database=database,
        datasets=datasets,
        orbit_outliers=orbit_outliers,
        propagator=propagator,
        propagator_kwargs=propagator_kwargs,
    )

    database.frames.close()

    return ipod_orbits, ipod_orbits_members, ipod_precovery_candidates, ipod_summary


ipod_worker_remote.options(num_cpus=1, num_returns=1)


def iterative_precovery_and_differential_correction(
    orbits: Union[FittedOrbits, ray.ObjectRef],
    orbit_members: Optional[Union[FittedOrbitMembers, ray.ObjectRef]] = None,
    observations: Optional[Union[Observations, ray.ObjectRef]] = None,
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
    database_directory: str = "",
    datasets: Optional[set[str]] = None,
    orbit_outliers: Optional[OrbitOutliers] = None,
    propagator: Type[Propagator] = PYOORB,
    propagator_kwargs: dict = {},
    chunk_size: int = 10,
    max_processes: Optional[int] = 1,
) -> Tuple[FittedOrbits, FittedOrbitMembers, PrecoveryCandidates, SearchSummary]:
    """
    Perform iterative precovery and differential correction on the input orbits.

    Parameters
    ----------
    orbits : Union[FittedOrbits, ray.ObjectRef]
        The orbits to perform iterative precovery and differential correction on.
    orbit_members : Optional[Union[FittedOrbitMembers, ray.ObjectRef]]
        The orbit members to include in the iterative precovery and differential correction.
        If none are provided then the orbit members will be constructed from the first
        iteration of precovery.
    observations : Optional[Union[Observations, ray.ObjectRef]]
        The observations from which orbit_members are derived. If orbit_members are not provided
        then these observations will be ignored.


        ...

    """
    time_start = time.perf_counter()
    logger.info("Running iterative precovery and differential correction...")
    if isinstance(orbits, ray.ObjectRef):
        orbits_ref = orbits
        orbits = ray.get(orbits)
        logger.info("Retrieved orbits from the object store.")
    else:
        orbits_ref = None

    if isinstance(orbit_members, ray.ObjectRef):
        orbit_members_ref = orbit_members
        orbit_members = ray.get(orbit_members)
        logger.info("Retrieved orbit members from the object store.")
    else:
        orbit_members_ref = None

    if isinstance(observations, ray.ObjectRef):
        observations_ref = observations
        observations = ray.get(observations)
        logger.info("Retrieved observations from the object store.")
    else:
        observations_ref = None

    if len(orbits) == 0:
        logger.info("Received no orbits or orbit members.")
        od_orbits = FittedOrbits.empty()
        od_orbit_members = FittedOrbitMembers.empty()
        time_end = time.perf_counter()
        logger.info(f"Differentially corrected {len(od_orbits)} orbits.")
        logger.info(
            f"Differential correction completed in {time_end - time_start:.3f} seconds."
        )
        return (
            od_orbits,
            od_orbit_members,
            PrecoveryCandidates.empty(),
            SearchSummary.empty(),
        )

    orbit_ids = orbits.orbit_id.to_numpy(zero_copy_only=False)

    ipod_orbits = FittedOrbits.empty()
    ipod_orbit_members = FittedOrbitMembers.empty()
    ipod_precovery_candidates = PrecoveryCandidates.empty()
    ipod_summary = SearchSummary.empty()

    if max_processes is None:
        max_processes = mp.cpu_count()

    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:
        refs_to_free = []

        orbit_ids_ref = ray.put(orbit_ids)
        orbit_ids = ray.get(orbit_ids_ref)
        refs_to_free.append(orbit_ids_ref)
        logger.info("Placed orbit IDs in the object store.")

        if orbits_ref is None:
            orbits_ref = ray.put(orbits)
            orbits = ray.get(orbits_ref)
            refs_to_free.append(orbits_ref)
            logger.info("Placed orbits in the object store.")

        if orbit_members_ref is None and orbit_members is not None:
            orbit_members_ref = ray.put(orbit_members)
            orbit_members = ray.get(orbit_members_ref)
            refs_to_free.append(orbit_members_ref)
            logger.info("Placed orbit members in the object store.")

        if observations_ref is None and observations is not None:
            observations_ref = ray.put(observations)
            refs_to_free.append(observations_ref)
            observations = ray.get(observations_ref)
            logger.info("Placed observations in the object store.")

        chunk_size = np.minimum(
            np.ceil(len(orbit_ids) / max_processes).astype(int), chunk_size
        )
        logger.info(
            f"Distributing orbits in chunks of {chunk_size} to {max_processes} workers."
        )

        futures = []
        for orbit_ids_indices in _iterate_chunk_indices(orbit_ids, chunk_size):
            futures.append(
                ipod_worker_remote.remote(
                    orbit_ids_ref,
                    orbit_ids_indices,
                    orbits_ref,
                    orbit_members_ref,
                    observations_ref,
                    min_tolerance=min_tolerance,
                    max_tolerance=max_tolerance,
                    tolerance_step=tolerance_step,
                    delta_time=delta_time,
                    rchi2_threshold=rchi2_threshold,
                    outlier_chi2=outlier_chi2,
                    reconsider_chi2=reconsider_chi2,
                    min_mjd=min_mjd,
                    max_mjd=max_mjd,
                    astrometric_errors=astrometric_errors,
                    database_directory=database_directory,
                    datasets=datasets,
                    orbit_outliers=orbit_outliers,
                    propagator=propagator,
                    propagator_kwargs=propagator_kwargs,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                (
                    ipod_orbits_chunk,
                    ipod_orbit_members_chunk,
                    ipod_precovery_candidates_chunk,
                    ipod_summary_chunk,
                ) = ray.get(finished[0])

                ipod_orbits = qv.concatenate([ipod_orbits, ipod_orbits_chunk])
                if ipod_orbits.fragmented():
                    ipod_orbits = qv.defragment(ipod_orbits)

                ipod_orbit_members = qv.concatenate(
                    [ipod_orbit_members, ipod_orbit_members_chunk]
                )
                if ipod_orbit_members.fragmented():
                    ipod_orbit_members = qv.defragment(ipod_orbit_members)

                ipod_precovery_candidates = qv.concatenate(
                    [ipod_precovery_candidates, ipod_precovery_candidates_chunk]
                )
                if ipod_precovery_candidates.fragmented():
                    ipod_precovery_candidates = qv.defragment(ipod_precovery_candidates)

                ipod_summary = qv.concatenate([ipod_summary, ipod_summary_chunk])
                if ipod_summary.fragmented():
                    ipod_summary = qv.defragment(ipod_summary)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            (
                ipod_orbits_chunk,
                ipod_orbit_members_chunk,
                ipod_precovery_candidates_chunk,
                ipod_summary_chunk,
            ) = ray.get(finished[0])

            ipod_orbits = qv.concatenate([ipod_orbits, ipod_orbits_chunk])
            if ipod_orbits.fragmented():
                ipod_orbits = qv.defragment(ipod_orbits)

            ipod_orbit_members = qv.concatenate(
                [ipod_orbit_members, ipod_orbit_members_chunk]
            )
            if ipod_orbit_members.fragmented():
                ipod_orbit_members = qv.defragment(ipod_orbit_members)

            ipod_precovery_candidates = qv.concatenate(
                [ipod_precovery_candidates, ipod_precovery_candidates_chunk]
            )
            if ipod_precovery_candidates.fragmented():
                ipod_precovery_candidates = qv.defragment(ipod_precovery_candidates)

            ipod_summary = qv.concatenate([ipod_summary, ipod_summary_chunk])
            if ipod_summary.fragmented():
                ipod_summary = qv.defragment(ipod_summary)

        if len(refs_to_free) > 0:
            ray.internal.free(refs_to_free)
            logger.info(
                f"Removed {len(refs_to_free)} references from the object store."
            )

    else:
        for orbit_ids_chunk in _iterate_chunks(orbit_ids, chunk_size):
            (
                ipod_orbits_chunk,
                ipod_orbit_members_chunk,
                ipod_precovery_candidates_chunk,
                ipod_summary_chunk,
            ) = ipod_worker(
                orbit_ids_chunk,
                orbits,
                orbit_members,
                observations,
                min_tolerance=min_tolerance,
                max_tolerance=max_tolerance,
                tolerance_step=tolerance_step,
                delta_time=delta_time,
                rchi2_threshold=rchi2_threshold,
                outlier_chi2=outlier_chi2,
                reconsider_chi2=reconsider_chi2,
                min_mjd=min_mjd,
                max_mjd=max_mjd,
                astrometric_errors=astrometric_errors,
                database=database_directory,
                datasets=datasets,
                orbit_outliers=orbit_outliers,
                propagator=propagator,
                propagator_kwargs=propagator_kwargs,
            )
            ipod_orbits = qv.concatenate([ipod_orbits, ipod_orbits_chunk])
            if ipod_orbits.fragmented():
                ipod_orbits = qv.defragment(ipod_orbits)
            ipod_orbit_members = qv.concatenate(
                [ipod_orbit_members, ipod_orbit_members_chunk]
            )
            if ipod_orbit_members.fragmented():
                ipod_orbit_members = qv.defragment(ipod_orbit_members)
            ipod_precovery_candidates = qv.concatenate(
                [ipod_precovery_candidates, ipod_precovery_candidates_chunk]
            )
            if ipod_precovery_candidates.fragmented():
                ipod_precovery_candidates = qv.defragment(ipod_precovery_candidates)

            ipod_summary = qv.concatenate([ipod_summary, ipod_summary_chunk])
            if ipod_summary.fragmented():
                ipod_summary = qv.defragment(ipod_summary)

    time_end = time.perf_counter()
    logger.info(
        f"Iteratively precovered and differentially corrected {len(ipod_orbits)} orbits."
    )
    logger.info(
        f"Iterative precovery and differential correction completed in {time_end - time_start:.3f} seconds."
    )

    return ipod_orbits, ipod_orbit_members, ipod_precovery_candidates, ipod_summary
