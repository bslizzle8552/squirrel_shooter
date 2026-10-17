from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from squirrel_shooter.collector_control_config import (
    AutomaticFireConfig,
    PhysicalControlConfig,
    parse_automatic_fire,
    parse_physical_control,
)
from squirrel_shooter.pan_tilt import PanTiltConfig


def test_missing_sections_leave_all_physical_actions_disabled():
    physical = parse_physical_control()
    automatic = parse_automatic_fire()
    assert not physical.enabled and not physical.valve_enabled
    assert not automatic.enabled and automatic.dry_run
    assert automatic.minimum_confidence is None
    assert physical.calibration_file == Path("collector-calibration.json")
    with pytest.raises(FrozenInstanceError):
        physical.enabled = True


def test_historical_hardware_defaults_are_preserved_without_enabling_them():
    physical = PhysicalControlConfig()
    angles = physical.pan_tilt
    assert (angles.i2c_address, angles.pan_channel, angles.tilt_channel) == (0x40, 0, 1)
    assert (angles.pan_pulse_min_us, angles.pan_pulse_max_us) == (600, 2400)
    assert (angles.tilt_pulse_min_us, angles.tilt_pulse_max_us) == (600, 2400)
    assert (angles.pan_min, angles.pan_center, angles.pan_max) == (30, 85, 150)
    assert (angles.tilt_min, angles.tilt_center, angles.tilt_max) == (70, 85, 150)
    assert (angles.park_pan, angles.park_tilt) == (85, 82)
    assert (physical.valve_gpio_pin, physical.valve_active_high) == (24, True)
    assert physical.pulse_seconds == physical.maximum_pulse_seconds == .40
    assert physical.cooldown_seconds == 10


@pytest.mark.parametrize("field", ["enabled", "valve_enabled", "valve_active_high"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_physical_boolean_fields_reject_truthy_coercion(field, value):
    with pytest.raises(ValueError, match=field):
        parse_physical_control({field: value})


@pytest.mark.parametrize("field", ["enabled", "dry_run"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_automatic_boolean_fields_reject_truthy_coercion(field, value):
    with pytest.raises(ValueError, match=field):
        parse_automatic_fire({field: value})


@pytest.mark.parametrize("section, parser", [
    ({"enable": True}, parse_physical_control),
    ({"minimum_confidence": .2}, parse_physical_control),
    ({"pan_tilt": {"pan_centerr": 90}}, parse_physical_control),
    ({"min_confidence": .2}, parse_automatic_fire),
])
def test_misspelled_or_misplaced_fields_fail_closed(section, parser):
    with pytest.raises(ValueError, match="Unknown"):
        parser(section)


@pytest.mark.parametrize("raw", [True, [], "enabled", {1: True}])
def test_non_mapping_sections_and_non_string_keys_are_rejected(raw):
    for parser in (parse_physical_control, parse_automatic_fire):
        with pytest.raises(ValueError, match="mapping"):
            parser(raw)


@pytest.mark.parametrize("raw", [None, True, [], "default"])
def test_explicit_invalid_pan_tilt_section_is_rejected(raw):
    with pytest.raises(ValueError, match="pan_tilt"):
        parse_physical_control({"pan_tilt": raw})


@pytest.mark.parametrize("field", ["pulse_seconds", "maximum_pulse_seconds"])
@pytest.mark.parametrize("value", [False, 0, -.1, float("nan"), float("inf"), .400001, 3])
def test_pulse_cannot_exceed_hard_ceiling_or_use_invalid_duration(field, value):
    with pytest.raises(ValueError, match=field):
        parse_physical_control({field: value})


def test_lower_configured_pulse_ceiling_is_enforced():
    physical = parse_physical_control({"pulse_seconds": .25, "maximum_pulse_seconds": .25})
    assert physical.pulse_seconds == physical.maximum_pulse_seconds == .25
    with pytest.raises(ValueError, match="pulse_seconds"):
        replace(physical, pulse_seconds=.3)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 3600.01])
def test_cooldown_requires_a_finite_positive_bounded_duration(value):
    with pytest.raises(ValueError, match="cooldown_seconds"):
        PhysicalControlConfig(cooldown_seconds=value)


@pytest.mark.parametrize("value", [0, 1.001, float("nan"), float("inf"), True])
def test_settle_time_cannot_be_zero_unbounded_or_invalid(value):
    with pytest.raises(ValueError, match="settling_delay_seconds"):
        parse_physical_control({"pan_tilt": {"settling_delay_seconds": value}})


@pytest.mark.parametrize("changes", [
    {"pan_min": -1}, {"pan_max": 181}, {"tilt_min": -1}, {"tilt_max": 181},
    {"park_pan": 151}, {"pan_channel": 1}, {"i2c_address": True}, {"i2c_address": "0x40"},
])
def test_servo_geometry_channels_and_i2c_reject_invalid_configuration(changes):
    with pytest.raises(ValueError):
        parse_physical_control({"pan_tilt": changes})


@pytest.mark.parametrize("value", [None, True, -1, 28, 24.0, "24"])
def test_valve_requires_explicit_valid_bcm_integer(value):
    with pytest.raises(ValueError, match="valve_gpio_pin"):
        PhysicalControlConfig(valve_gpio_pin=value)


def test_active_low_valve_is_not_an_accepted_wiring_override():
    with pytest.raises(ValueError, match="GPIO LOW"):
        PhysicalControlConfig(valve_active_high=False)


@pytest.mark.parametrize("path", ["", " ", ".", None, False])
def test_calibration_path_must_name_a_file(path):
    with pytest.raises(ValueError, match="calibration_file"):
        parse_physical_control({"calibration_file": path})


def test_parser_leaves_relative_calibration_path_for_config_loader():
    physical = parse_physical_control({"calibration_file": "field/calibration.json",
                                       "pan_tilt": {"settling_delay_seconds": .2}})
    assert physical.calibration_file == Path("field/calibration.json")
    assert not physical.calibration_file.is_absolute()
    assert physical.pan_tilt.settling_delay_seconds == .2
    assert physical.pan_tilt.pan_center == 85


@pytest.mark.parametrize("dry_run", [True, False])
def test_enabled_automatic_policy_always_requires_explicit_confidence(dry_run):
    with pytest.raises(ValueError, match="explicitly set"):
        AutomaticFireConfig(enabled=True, dry_run=dry_run)


@pytest.mark.parametrize("value", [True, -.1, 1.01, "0.10", float("nan"), float("inf")])
def test_automatic_confidence_rejects_invalid_numeric_values(value):
    with pytest.raises(ValueError, match="minimum_confidence"):
        parse_automatic_fire({"minimum_confidence": value})


def test_explicit_automatic_confidence_is_independent_of_recording_policy():
    automatic = parse_automatic_fire({"enabled": True, "dry_run": True, "minimum_confidence": .65})
    assert automatic.minimum_confidence == .65
    assert automatic.enabled and automatic.dry_run
    # Physical master/valve enablement and relative path resolution belong to load_config.
    assert not PhysicalControlConfig().enabled


def test_direct_dataclass_construction_does_not_accept_a_pan_tilt_mapping():
    with pytest.raises(ValueError, match="PanTiltConfig"):
        PhysicalControlConfig(pan_tilt={})
    assert isinstance(parse_physical_control({}).pan_tilt, PanTiltConfig)


@pytest.mark.parametrize("raw", [
    {"pulse_seconds": 10**1000},
    {"pan_tilt": {"movement_speed": 10**1000}},
])
def test_extreme_numeric_configuration_is_a_validation_error(raw):
    with pytest.raises(ValueError):
        parse_physical_control(raw)
