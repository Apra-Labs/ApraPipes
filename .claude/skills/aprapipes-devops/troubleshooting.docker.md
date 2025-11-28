# Docker/WSL Build Troubleshooting

Platform-specific troubleshooting for Docker containerized builds and WSL environments.

**Scope**: Docker containers and Windows Subsystem for Linux (WSL) builds.

---

## Docker-Specific Architecture

### Build Configuration
- **Environment**: Docker container
- **Base Images**: Ubuntu, Alpine, or custom
- **Volumes**: Code mounted, caches mounted
- **Networking**: Container network access for vcpkg downloads

### Key Characteristics
- Isolated environment (clean state each run)
- Volume mounts for code and caches
- Network access required for vcpkg
- User permissions in containers

---

## Issue D1: Cache Mounting in Containers

**Symptom**:
```
vcpkg cache not persisting between builds
downloads re-fetched every time
```

**Root Cause**:
- vcpkg cache not mounted as volume
- Container destroyed after build, cache lost

**Fix**:

Mount vcpkg cache directory:
```yaml
# docker-compose.yml or docker run command
volumes:
  - ./vcpkg:/workspace/vcpkg
  - vcpkg-cache:/root/.cache/vcpkg  # Persist cache
```

**To Be Expanded**: Add specific Docker volume configuration examples.

---

## Issue D2: Container Cleanup Between Builds

**Symptom**:
- Old build artifacts remain
- Disk space issues in container

**Fix**:

Ensure containers are removed after each build:
```bash
docker-compose down -v  # Remove volumes
docker system prune -af  # Clean up
```

---

## Issue D3: Volume Permissions

**Symptom**:
```
Permission denied writing to /workspace
cannot create directory
```

**Root Cause**:
- User ID mismatch between host and container
- Volume mounted with wrong permissions

**Fix**:

Match container user to host user:
```dockerfile
RUN useradd -u 1000 -m builder
USER builder
```

Or run container with host user:
```bash
docker run --user $(id -u):$(id -g) ...
```

---

## WSL-Specific Issues

### Issue WSL1: Path Translation

**Symptom**:
```
error: cannot find file at /mnt/c/...
Windows path not accessible from WSL
```

**Root Cause**:
- Path format differences between Windows and WSL
- Symbolic links not working across boundary

**Fix**:

Use WSL paths consistently:
```bash
# Instead of: /mnt/c/Users/...
# Use: /home/user/...
```

---

### Issue WSL2: Network Access

**Symptom**:
```
vcpkg download failed
cannot connect to github.com
```

**Root Cause**:
- WSL2 uses virtualized network
- Firewall blocking WSL2 network

**Fix**:

Check WSL network configuration:
```bash
# Test connectivity
ping github.com
curl -I https://github.com
```

---

## Docker/WSL Quick Fixes Checklist

### Docker Checklist
- [ ] vcpkg cache mounted as volume
- [ ] Code directory mounted correctly
- [ ] Network access available
- [ ] User permissions correct
- [ ] Container cleanup after builds

### WSL Checklist
- [ ] Paths use WSL format (not /mnt/c/...)
- [ ] Network connectivity works
- [ ] vcpkg downloads successful
- [ ] File permissions correct

---

## To Be Expanded

This guide will be expanded as Docker/WSL-specific issues are encountered:
- Dockerfile examples for ApraPipes builds
- docker-compose configuration for multi-stage builds
- WSL2 performance optimization
- Container registry integration
- Multi-platform Docker builds (x64 + ARM64)

**Cross-Platform Issues**: See other troubleshooting guides for platform-specific issues within containers.

---

**Last Updated**: 2024-11-28
**Status**: Outline - expand as issues occur
**Applies to**: Docker containerized builds and WSL environments
**Related Guides**: reference.md, troubleshooting.linux.md
