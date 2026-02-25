#!/bin/bash

# This script is used to tag a new version of the software.
echo "Script $0 started"

export SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPTS_DIR}/init.sh

print_help() {
    echo "*****************************************************************************"
    echo "Usage: $0 -v <version_number> [-h]"
    echo "  -v | --version-number  version number to tag the new version, like v2.1.0"
    echo "  -h | --help            print this help message"
    echo "*****************************************************************************"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--version-number) versionNumber="$2"; shift 2;;
        -h|--help) print_help;;
        *) shift;;
    esac
done

if [ -z "$versionNumber" ]
then
    print_help
fi

# If not in the main branch, print a warning TODO add more severe handling, require input to continue
currentBranch=$(git branch --show-current)
if [ "$currentBranch" != "main" ] && [ "$currentBranch" != "master" ]
then
    echo "WARNING: You are not in the main branch. You are in $currentBranch branch."
fi

# pull, just to make sure to have all remote changes
git pull

# Run smoke tests before any version/tag changes
echo "Running smoke tests before tagging..."
test_command=". ${REPO_DIR}/test/testAllApps.sh"
echo "$test_command"
$test_command
if [ $? -ne 0 ]
then
    echo "ERROR: tests failed. Aborting tagging workflow."
    exit 1
fi

# Update docs/version.txt after tests pass
version_file="${REPO_DIR}/docs/version.txt"
echo "$versionNumber" > $version_file
git add $version_file
git commit -m "Update version number to $versionNumber"
git push

# Tag the new version
git tag -a $versionNumber

# here there should be nano opening to write the message

git push origin $versionNumber

echo "Tagged version $versionNumber successfully"

exit 0