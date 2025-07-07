@echo off
setlocal EnableDelayedExpansion

rem Define arrays for environment names, algorithms, policies, and update types
set "envs=RiverSwim Chain Loop Gridworld SixArms ThreeArms Taxi KnightQuest"
set "algos=boot-ql"
set "policies=boot weighted"
set "update_types=weighted"

rem Loop through each combination of parameters
for %%a in (%envs%) do (
    for %%d in (%algos%) do (
        for %%b in (%policies%) do (
            for %%c in (%update_types%) do (
                echo Running script with name=%%a, algorithm=%%d, policy=%%b, and update type=%%c
                python run_other_algos.py -name "%%a" -algo "%%d" -policy "%%b" -update_type "%%c"
                echo --------------------------------------------------------
            )
        )
    )
)

endlocal