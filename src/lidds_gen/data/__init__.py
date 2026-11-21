"""Recording-level preprocessing and experiment assembly (no torch dependency)."""

from .recording import iter_scenario_recordings, load_recording
from .splits import RecordingInfo, resplit_recordings, select_setting
from .store import ScenarioWindowWriter
from .vocab import Vocabulary
from .windows import windows_for_recording

SCENARIOS = (
    "Bruteforce_CWE-307", "CVE-2012-2122", "CVE-2014-0160", "CVE-2017-7529",
    "CVE-2018-3760", "CVE-2019-5418", "CVE-2020-9484", "CWE-89-SQL-injection",
    "PHP_CWE-434", "EPS_CWE-434", "ZipSlip", "CVE-2017-12635_6",
    "CVE-2020-13942", "CVE-2020-23839", "Juice-Shop",
)

__all__ = [
    "SCENARIOS", "RecordingInfo", "Vocabulary", "ScenarioWindowWriter",
    "iter_scenario_recordings", "load_recording", "resplit_recordings",
    "select_setting", "windows_for_recording",
]
