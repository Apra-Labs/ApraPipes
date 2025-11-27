#!/bin/bash
# Script to swap to minimal vcpkg.json for fast testing

if [ ! -f "base/vcpkg.json.full-backup" ]; then
    echo "Backing up full vcpkg.json..."
    cp base/vcpkg.json base/vcpkg.json.full-backup
fi

echo "Switching to minimal vcpkg.json (glib only)..."
cp base/vcpkg.json.minimal-test base/vcpkg.json

echo "Minimal vcpkg.json activated. This will test:"
echo "  - libxml2 hash (dependency of glib)"
echo "  - Python distutils (required by glib build)"
