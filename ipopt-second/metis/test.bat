@echo off

setlocal
pushd %1

set temp_folder="%~dp0./tmp/" || goto :FINALLY
if exist %temp_folder% rd /q /s %temp_folder% || goto :FINALLY

conan source . --source-folder=tmp/source || goto :FINALLY
conan install . --install-folder=tmp/install || goto :FINALLY
conan build . --source-folder=tmp/source --install-folder=tmp/install --build-folder=tmp/build || goto :FINALLY
conan package . --source-folder=tmp/source --install-folder=tmp/install --build-folder=tmp/build --package-folder=tmp/package || goto :FINALLY
conan export-pkg . user/testing --source-folder=tmp/source --install-folder=tmp/install --build-folder=tmp/build -f || goto :FINALLY

conan export-pkg . metis/5.1.0@tuncb/pangea --force || goto :FINALLY
conan test ./test_package metis/5.1.0@tuncb/pangea || goto :FINALLY

:FINALLY
  popd
  endlocal

  IF /I "%ERRORLEVEL%" NEQ "0" (
      echo Solution generation failed with error #%ERRORLEVEL%.
      exit /b %ERRORLEVEL%
  )