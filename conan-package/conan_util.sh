alias co_s='conan source . --source-folder=tmp/source'
alias co_i='conan install . --install-folder=tmp/build'
alias co_b='CONAN_CPU_COUNT=10 conan build . --source-folder=tmp/source --build-folder=tmp/build'
alias co_p='conan package . --source-folder=tmp/source --build-folder=tmp/build --package-folder=tmp/package'
alias co_e='conan export-pkg . curt/testing --package-folder=tmp/package'
#conan test test_package selene/0.3.1@curt/testing
#conan create . curt/testing
