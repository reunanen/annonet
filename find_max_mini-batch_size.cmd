@REM USAGE (for example):
@REM annonet_train_cuda training_data_dir --relative-training-length=2.0 -c 10000 --class-weight=0.0 --image-weight=0.0 --net-width-scaler=2.0

@SET /A MIN_MINIBATCH_SIZE = 2
@SET /A MAX_MINIBATCH_SIZE = 0
@SET /A CURRENT_MINIBATCH_SIZE = 100
@SET /A TEST_STEP_COUNT = 3

@REM increase this parameter to accept a size that is smaller than "optimal"
@REM (and consequently proceed to the actual run a little faster)
@SET /A TOLERANCE = 0

@ECHO.

@GOTO TEST

:INCREASE_MINIBATCH_SIZE_BY_FACTOR_OF_2
@SET /A NEW_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE% * 2
@GOTO MAYBE_INCREASE_MINIBATCH_SIZE

:INCREASE_MINIBATCH_SIZE_HALFWAY
@SET /A NEW_MINIBATCH_SIZE = (%CURRENT_MINIBATCH_SIZE% + %MAX_MINIBATCH_SIZE% + 1) / 2
@GOTO MAYBE_INCREASE_MINIBATCH_SIZE

:MAYBE_INCREASE_MINIBATCH_SIZE
@IF %MAX_MINIBATCH_SIZE% EQU 0 GOTO INCREASE_MINIBATCH_SIZE
@SET /A CURRENT_MINIBATCH_SIZE_INCLUDING_TOLERANCE = %CURRENT_MINIBATCH_SIZE% + %TOLERANCE%
@IF %CURRENT_MINIBATCH_SIZE_INCLUDING_TOLERANCE% GEQ %MAX_MINIBATCH_SIZE% GOTO ACTUAL_RUN

:INCREASE_MINIBATCH_SIZE
@SET /A CURRENT_MINIBATCH_SIZE = %NEW_MINIBATCH_SIZE%
@GOTO TEST

:DECREASE_MINIBATCH_SIZE
@SET /A MAX_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE% - 1
@SET /A CURRENT_MINIBATCH_SIZE = (%MIN_MINIBATCH_SIZE% + %CURRENT_MINIBATCH_SIZE%) / 2
@IF %CURRENT_MINIBATCH_SIZE% EQU %MIN_MINIBATCH_SIZE% GOTO ACTUAL_RUN
@SET /A MIN_MINIBATCH_SIZE_INCLUDING_TOLERANCE = %MIN_MINIBATCH_SIZE% + %TOLERANCE%
@IF %MAX_MINIBATCH_SIZE% GEQ %MIN_MINIBATCH_SIZE_INCLUDING_TOLERANCE% GOTO TEST
@SET /A CURRENT_MINIBATCH_SIZE = %MIN_MINIBATCH_SIZE%
@GOTO ACTUAL_RUN

:TEST
@ECHO Trying mini-batch size %CURRENT_MINIBATCH_SIZE%

@REM make the test call
@%* -b %CURRENT_MINIBATCH_SIZE% --max-total-steps=%TEST_STEP_COUNT% 1> find_max_mini-batch_size_test_log_1.txt 2> find_max_mini-batch_size_test_log_2.txt

@IF ERRORLEVEL 1 GOTO DECREASE_MINIBATCH_SIZE
@SET /A MIN_MINIBATCH_SIZE = %CURRENT_MINIBATCH_SIZE%
@IF %MAX_MINIBATCH_SIZE% EQU 0 GOTO INCREASE_MINIBATCH_SIZE_BY_FACTOR_OF_2
@GOTO INCREASE_MINIBATCH_SIZE_HALFWAY

@:DECREASE_MINIBATCH_SIZE_BY_ONE
@SET /A CURRENT_MINIBATCH_SIZE -= 1
@ECHO.
@ECHO Decreased mini-batch size by one, now %CURRENT_MINIBATCH_SIZE%

:ACTUAL_RUN

@REM make the actual call
%* -b %CURRENT_MINIBATCH_SIZE%

@IF ERRORLEVEL 1 GOTO DECREASE_MINIBATCH_SIZE_BY_ONE
