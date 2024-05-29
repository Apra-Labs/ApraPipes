#!/bin/bash

# Get command line arguments
removeCUDA=$1
removeOpenCV=$2
onlyOpenCV=$3

# Load JSON file and print its contents
fileName="base/vcpkg.json"
v=$(cat "$fileName" | jq -c '.')
echo "Original JSON file contents:"
echo "$v"

# Process command line arguments
if $removeCUDA; then
    echo "Removing CUDA..."
    # Loop through each "opencv4" instance
    for index in $(echo "$v" | jq -r '.dependencies | keys | .[]'); do
        name=$(echo "$v" | jq -r ".dependencies[$index].name")
        if [ "$name" == "opencv4" ]; then
            # Remove "cuda" and "cudnn" features for this "opencv4" instance
            v=$(echo "$v" | jq ".dependencies[$index].features |= map(select(. != \"cuda\" and . != \"cudnn\"))")
        fi
        if [ "$name" == "whisper"]; then
            # Remove "cuda" features for this "whisper" instance
            v=$(echo "$v" | jq ".dependencies[$index].features |= map(select(. != \"cuda\"))")
        fi
        if [ "$name" == "llama"]; then
            # Remove "cuda" features for this "llama" instance
            v=$(echo "$v" | jq ".dependencies[$index].features |= map(select(. != \"cuda\"))")
        fi
    done
fi

if $removeOpenCV; then
    echo "Removing OpenCV..."
    count=$(echo "$v" | jq '.dependencies | length')
    for ((index=count-1; index >= 0; index--)); do
        name=$(echo "$v" | jq -r ".dependencies[$index].name")
        if [ "$name" == "opencv4" ]; then
            # Remove the entire "opencv4" instance
            v=$(echo "$v" | jq "del(.dependencies[$index])")
        fi
    done
fi

if $onlyOpenCV; then
    echo "Keeping only OpenCV..."
    # Loop through each dependency and remove other dependencies except "opencv4"
    count=$(echo "$v" | jq '.dependencies | length')
    for ((index = count - 1; index >= 0; index--)); do
        name=$(echo "$v" | jq -r ".dependencies[$index].name")
        if [ "$name" != "opencv4" ]; then
            # Remove the entire dependency
            v=$(echo "$v" | jq "del(.dependencies[$index])")
        fi
    done
fi

# Save the modified object back to the JSON file
echo "Modified JSON file contents:"
echo "$v" | jq .
echo "$v" | jq . > "$fileName"

# Print all command line arguments
echo "Command line arguments:"
echo "$@"