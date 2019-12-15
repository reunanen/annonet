REM USAGE (for example):
REM annonet_train_cuda training_data_dir --relative-training-length=2.0 -c 10000 --class-weight=0.0 --image-weight=0.0 --net-width-scaler=2.0

SET /A MIN_MINIBATCH_SIZE = 2
SET /A MAX_MINIBATCH_SIZE = 0
SET /A CURRENT_MINIBATCH_SIZE = 100
SET /A TEST_STEP_COUNT = 4

GOTO TEST

:INCREASE_MINIBATCH_SIZE_BY_FACTOR_OF_2
SET /A MIN_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE%
SET /A CURRENT_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE% * 2
GOTO TEST

:INCREASE_MINIBATCH_SIZE_HALFWAY
SET /A MIN_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE%
SET /A CURRENT_MINIBATCH_SIZE = (%CURRENT_MINIBATCH_SIZE% + %MAX_MINIBATCH_SIZE%) / 2
GOTO TEST

:DECREASE_MINIBATCH_SIZE
SET /A MAX_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE% - 1
SET /A CURRENT_MINIBATCH_SIZE = (%MIN_MINIBATCH_SIZE% + %CURRENT_MINIBATCH_SIZE%) / 2
GOTO TEST

:TEST
REM make the test call
%* -b %CURRENT_MINIBATCH_SIZE% --max-total-steps=%TEST_STEP_COUNT%

IF ERRORLEVEL 1 GOTO DECREASE_MINIBATCH_SIZE
IF %MAX_MINIBATCH_SIZE% EQU 0 GOTO INCREASE_MINIBATCH_SIZE_BY_FACTOR_OF_2
SET /A MIN_MINIBATCH_SIZE_PLUS_1 = %MIN_MINIBATCH_SIZE% + 1
IF %MIN_MINIBATCH_SIZE_PLUS_1% GEQ %MAX_MINIBATCH_SIZE% GOTO ACTUAL_RUN
GOTO INCREASE_MINIBATCH_SIZE_HALFWAY

:DECREASE_MINIBATCH_SIZE_BY_ONE
SET /A CURRENT_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE% - 1

:ACTUAL_RUN

REM make the real call
%* -b %CURRENT_MINIBATCH_SIZE%

IF ERRORLEVEL 1 GOTO DECREASE_MINIBATCH_SIZE_BY_ONE
