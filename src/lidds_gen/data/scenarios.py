"""The 15 LID-DS-2021 scenarios (Table 1 of the proposal).

Names must match the folder names of the extracted dataset.
"""

SINGLE_STEP_SCENARIOS = (
    "Bruteforce_CWE-307",
    "CVE-2012-2122",
    "CVE-2014-0160",
    "CVE-2017-7529",
    "CVE-2018-3760",
    "CVE-2019-5418",
    "CVE-2020-9484",
    "CWE-89-SQL-injection",
    "PHP_CWE-434",
    "EPS_CWE-434",
    "ZipSlip",
)

MULTI_STEP_SCENARIOS = (
    "CVE-2017-12635_6",
    "CVE-2020-13942",
    "CVE-2020-23839",
    "Juice-Shop",
)

SCENARIOS = SINGLE_STEP_SCENARIOS + MULTI_STEP_SCENARIOS

# Official LID-DS-2021 partitions, relative to a scenario folder.
OFFICIAL_PARTITIONS = {
    "training": "training",
    "validation": "validation",
    "test_normal": "test/normal",
    "test_attack": "test/normal_and_attack",
}
