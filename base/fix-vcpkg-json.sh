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
    opencv=$(echo "$v" | jq '.dependencies[] | select(type == "string" or .name == "opencv4")')
    opencv=$(echo "$opencv" | jq '.features |= map(select(. != "cuda" and . != "cudnn"))')
    v=$(echo "$v" | jq '.dependencies[] |= select(type == "string" or .name != "opencv4")')
    v=$(echo "$v" | jq '.dependencies[] |= select(type != "null")')
    v=$(echo "$v" | jq ".dependencies += [$opencv]")
fi

if $removeOpenCV; then
    echo "Removing OpenCV..."
     v=$(echo "$v" | jq '.dependencies[] |= select(type == "string" or .name != "opencv4")')
     v=$(echo "$v" | jq '.dependencies[] |= select(type != "null")')
fi

if $onlyOpenCV; then
     echo "Keeping only OpenCV..."
     opencv=$(echo "$v" | jq '.dependencies[] | select(.name == "opencv4")')
     opencv=$(echo "$opencv" | jq '.features |= map(select(. != "cuda" and . != "cudnn"))')
     v=$(echo "$v" | jq '.dependencies = []')
     v=$(echo "$v" | jq ".dependencies += [$opencv]")
    
fi

 #Save the modified object back to the JSON file
 echo "Modified JSON file contents:"
 echo "$v" | jq .
 echo "$v" | jq . > "$fileName"

# Print all command line arguments
echo "Command line arguments:"
echo "$@"

