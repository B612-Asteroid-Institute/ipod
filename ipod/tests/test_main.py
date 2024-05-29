import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import quivr as qv
from adam_core.coordinates.covariances import CoordinateCovariances
from adam_core.coordinates.origin import Origin
from adam_core.coordinates.spherical import SphericalCoordinates
from adam_core.observers import Observers
from adam_core.orbit_determination.evaluate import (
    OrbitDeterminationObservations,
    evaluate_orbits,
)
from adam_core.propagator.adam_pyoorb import PYOORBPropagator as PYOORB
from adam_core.utils.helpers import make_observations, make_real_orbits
from precovery.ingest import index
from precovery.precovery_db import PrecoveryDatabase

from ipod.ipod import OrbitOutliers, ipod

OBJECT_IDS = [
    "594913 'Aylo'chaxnim (2020 AV2)",
    "163693 Atira (2003 CP20)",
    "(2010 TK7)",
    "3753 Cruithne (1986 TO)",
    "54509 YORP (2000 PH5)",
    "2063 Bacchus (1977 HB)",
    "1221 Amor (1932 EA1)",
    "433 Eros (A898 PA)",
    "3908 Nyx (1980 PA)",
    "434 Hungaria (A898 RB)",
    "1876 Napolitania (1970 BA)",
    "2001 Einstein (1973 EB)",
    "2 Pallas (A802 FA)",
    "6 Hebe (A847 NA)",
    "6522 Aci (1991 NQ)",
    "10297 Lynnejones (1988 RJ13)",
    "17032 Edlu (1999 FM9)",
    "202930 Ivezic (1998 SG172)",
    "911 Agamemnon (A919 FB)",
    "1143 Odysseus (1930 BH)",
    "1172 Aneas (1930 UA)",
    "3317 Paris (1984 KF)",
    "5145 Pholus (1992 AD)",
    "5335 Damocles (1991 DA)",
    "15760 Albion (1992 QB1)",
    "15788 (1993 SB)",
    "15789 (1993 SC)",
    "1I/'Oumuamua (A/2017 U1)",
]
TOLERANCES = {
    "default": 0.1 / 3600,
    "594913 'Aylo'chaxnim (2020 AV2)": 2 / 3600,
    "1I/'Oumuamua (A/2017 U1)": 5 / 3600,
}


@pytest.fixture
def observations():
    yield make_observations()


@pytest.fixture
def orbits():
    return make_real_orbits()


@pytest.fixture
def precovery_db(observations):

    exposures, detections, associations = observations
    exposures_df = exposures.to_dataframe()
    exposures_df["exposure_mjd_start"] = exposures.start_time.mjd().to_numpy()
    exposures_df["exposure_mjd_mid"] = exposures_df["exposure_mjd_start"] + (
        (exposures_df["duration"] / 86400) / 2
    )
    # getting issues with Rubin obscode plus pyoorb
    exposures_df = exposures_df[exposures_df["observatory_code"] != "X05"]

    detections_df = detections.to_dataframe()
    detections_df["mjd"] = detections.time.mjd().to_numpy()
    # join on exposure_id
    observations_df = detections_df.merge(
        exposures_df, left_on="exposure_id", right_on="id"
    )

    observations_df["obs_id"] = observations_df["id_x"]
    observations_df["dataset_id"] = "test"
    observations_df.rename(columns={"duration": "exposure_duration"}, inplace=True)

    # Add some offset data/noise to exposures, detections, associations
    offset_mask = range(0, len(observations_df), 6)
    # add an offset in ra/dec to the observations
    observations_df.loc[offset_mask, "ra"] += 0.1 / 3600
    observations_df.loc[offset_mask, "dec"] += 0.1 / 3600

    # add the ids of the adjusted observations to the offset_ids list
    offset_ids = observations_df.loc[offset_mask, "obs_id"].to_list()

    # Save to csv in a dataset directory for indexing
    data_dir = os.path.join(os.path.dirname(__file__), "data/index/test/")

    os.makedirs(data_dir, exist_ok=True)
    observations_df.to_csv(
        os.path.join(data_dir, "observations.csv"),
        index=False,
        columns=[
            "obs_id",
            "mjd",
            "ra",
            "dec",
            "ra_sigma",
            "dec_sigma",
            "mag_sigma",
            "filter",
            "mag",
            "observatory_code",
            "exposure_mjd_start",
            "exposure_mjd_mid",
            "exposure_duration",
            "dataset_id",
            "exposure_id",
        ],
    )

    test_db_dir = os.path.join(os.path.dirname(__file__), "data/db/")
    index(
        out_dir=test_db_dir,
        dataset_id="test",
        dataset_name="test",
        data_dir=data_dir,
        nside=16,
    )
    yield PrecoveryDatabase.from_dir(str(test_db_dir)), test_db_dir, offset_ids
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "data"))


@pytest.fixture
def od_observations(orbits, observations):
    exposures, detections, associations = observations

    od_obs = []
    for observatory_code in exposures.observatory_code.unique().to_pylist():
        # skip rubin for now
        if observatory_code == "X05":
            continue
        # Select the detections and exposures that match this observatory code

        exposures_i = exposures.apply_mask(
            pc.equal(exposures.observatory_code, observatory_code)
        )

        detections_i = detections.apply_mask(
            pc.is_in(detections.exposure_id, exposures_i.id)
        )
        observers = Observers.from_code(observatory_code, detections_i.time)
        # ovservations didn't come with sigmas, so we'll just fill some values
        sigmas = np.full((len(detections_i), 6), dtype="float64", fill_value=np.nan)
        sigmas[:, 1] = 0.1 / 3600
        sigmas[:, 2] = 0.1 / 3600
        coords = SphericalCoordinates.from_kwargs(
            lon=detections_i.ra,
            lat=detections_i.dec,
            covariance=CoordinateCovariances.from_sigmas(sigmas),
            origin=Origin.from_kwargs(
                code=np.full(len(detections_i), observatory_code, dtype="object")
            ),
            time=detections_i.time,
            frame="equatorial",
        )
        coords = coords.sort_by(
            [
                "time.days",
                "time.nanos",
                "origin.code",
            ]
        )
        observers = observers.sort_by(
            ["coordinates.time.days", "coordinates.time.nanos", "code"]
        )
        od_obs.append(
            OrbitDeterminationObservations.from_kwargs(
                id=detections_i.id,
                coordinates=coords,
                observers=observers,
            )
        )

    # concat the observatories
    od_obs = qv.concatenate(od_obs)
    yield od_obs


def test_index(precovery_db, orbits):
    db, test_db_dir, offset_ids = precovery_db
    # precover one of our orbits
    orbit = orbits[0]
    matches = [m for m in db.precover(orbit)]
    assert len(matches) > 0


def test_ipod_orbit_outliers(precovery_db, orbits, observations, od_observations):

    db, test_db_dir, offset_ids = precovery_db

    exposures, detections, associations = observations

    for object_id in OBJECT_IDS:

        # orbit = orbits.select("object_id", object_id)
        orbit = orbits.select("object_id", object_id)

        associations_i = associations.select("object_id", orbit.object_id[0])

        od_observations_i = od_observations.apply_mask(
            pc.is_in(od_observations.id, associations_i.detection_id)
        )

        # We are only fitting on a subset of the observations so that ipod has some new obs to find
        fitted_orbits, fitted_orbit_members = evaluate_orbits(
            orbit, od_observations_i[:10], propagator=PYOORB()
        )

        detections_i = detections.apply_mask(
            pc.is_in(detections.id, od_observations_i.id)
        )
        mjd_min = pc.min(detections_i.time.mjd())
        mjd_max = pc.max(detections_i.time.mjd())

        # Now let's create a set of orbit outliers
        outlier_detections = od_observations_i.apply_mask(
            pc.is_in(od_observations_i.id, pa.array(offset_ids))
        )
        orbit_outliers = OrbitOutliers.from_kwargs(
            orbit_id=pa.array(
                [orbit.orbit_id[0] for i in range(len(outlier_detections))]
            ),
            obs_id=outlier_detections.id,
        )

        ipod_result = ipod(
            fitted_orbits,
            od_observations_i[:10],
            max_tolerance=10.0,
            tolerance_step=2.0,
            delta_time=10.0,
            min_mjd=mjd_min.as_py() - 1.0,
            max_mjd=mjd_max.as_py() + 1.0,
            astrometric_errors={"default": (0.1, 0.1)},
            orbit_outliers=orbit_outliers,
            database=db,
        )

        (
            ipod_fitted_orbits_i,
            ipod_fitted_orbit_members_i,
            precovery_candidates,
            search_summary,
        ) = ipod_result
        # assert we got none of our outlier obs back
        assert (
            len(
                precovery_candidates.apply_mask(
                    pc.is_in(precovery_candidates.observation_id, orbit_outliers.obs_id)
                )
            )
            == 0
        )

        # assert that we got all the obs back save for the outliers
        all_obs_less_outliers = detections_i.apply_mask(
            pc.invert(pc.is_in(detections_i.id, orbit_outliers.obs_id))
        )
        np.testing.assert_equal(
            precovery_candidates.observation_id.to_numpy(zero_copy_only=False),
            all_obs_less_outliers.id.to_numpy(zero_copy_only=False),
        )
        break


def test_ipod_orbit_outliers_all_bad(
    precovery_db, orbits, observations, od_observations
):

    db, test_db_dir, offset_ids = precovery_db

    exposures, detections, associations = observations

    for object_id in OBJECT_IDS:

        # orbit = orbits.select("object_id", object_id)
        orbit = orbits.select("object_id", object_id)

        associations_i = associations.select("object_id", orbit.object_id[0])

        od_observations_i = od_observations.apply_mask(
            pc.is_in(od_observations.id, associations_i.detection_id)
        )

        # We are only fitting on a subset of the observations so that ipod has some new obs to find
        fitted_orbits, fitted_orbit_members = evaluate_orbits(
            orbit, od_observations_i[:10], propagator=PYOORB()
        )

        detections_i = detections.apply_mask(
            pc.is_in(detections.id, od_observations_i.id)
        )
        mjd_min = pc.min(detections_i.time.mjd())
        mjd_max = pc.max(detections_i.time.mjd())

        # Now let's create a set of orbit outliers
        outlier_detections = od_observations_i[:10]
        orbit_outliers = OrbitOutliers.from_kwargs(
            orbit_id=pa.array(
                [orbit.orbit_id[0] for i in range(len(outlier_detections))]
            ),
            obs_id=outlier_detections.id,
        )

        ipod_result = ipod(
            fitted_orbits,
            od_observations_i[:10],
            max_tolerance=10.0,
            tolerance_step=2.0,
            delta_time=10.0,
            min_mjd=mjd_min.as_py() - 1.0,
            max_mjd=mjd_max.as_py() + 1.0,
            astrometric_errors={"default": (0.1, 0.1)},
            orbit_outliers=orbit_outliers,
            database=db,
        )

        (
            ipod_fitted_orbits_i,
            ipod_fitted_orbit_members_i,
            precovery_candidates,
            search_summary,
        ) = ipod_result

        # assert we got none of our input obs back
        assert (
            len(
                precovery_candidates.apply_mask(
                    pc.is_in(
                        precovery_candidates.observation_id, od_observations_i[:10].id
                    )
                )
            )
            == 0
        )

        # assert that we got all the obs back save for the outliers
        all_obs_less_outliers = detections_i.apply_mask(
            pc.invert(pc.is_in(detections_i.id, od_observations_i[:10].id))
        )
        np.testing.assert_equal(
            precovery_candidates.observation_id.to_numpy(zero_copy_only=False),
            all_obs_less_outliers.id.to_numpy(zero_copy_only=False),
        )
        break
