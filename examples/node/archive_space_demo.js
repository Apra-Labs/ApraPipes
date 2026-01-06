/**
 * Archive Space Manager Demo
 *
 * This example demonstrates the ArchiveSpaceManager module which monitors
 * disk space usage and automatically deletes oldest files when thresholds
 * are exceeded.
 *
 * OBSERVABLE OUTPUT:
 * - Watch the terminal output to see files being created
 * - See the space manager detect when threshold is exceeded
 * - Observe files being deleted to bring space under control
 *
 * Run with: node archive_space_demo.js
 */

const fs = require('fs');
const path = require('path');

// Demo configuration
const DEMO_DIR = path.join(__dirname, 'archive_demo_data');
const FILE_SIZE = 1024 * 100; // 100KB per file
const NUM_FILES = 20;
const UPPER_WATERMARK = FILE_SIZE * 15; // Trigger cleanup at 1.5MB
const LOWER_WATERMARK = FILE_SIZE * 10; // Stop cleanup at 1MB

console.log('=== Archive Space Manager Demo ===\n');
console.log('This demo shows how ArchiveSpaceManager monitors disk usage');
console.log('and automatically deletes oldest files when thresholds are exceeded.\n');

// Setup demo directory
function setupDemoDir() {
    if (fs.existsSync(DEMO_DIR)) {
        fs.rmSync(DEMO_DIR, { recursive: true });
    }
    fs.mkdirSync(DEMO_DIR, { recursive: true });

    // Create subdirectory structure (simulates camera folders)
    const cam1Dir = path.join(DEMO_DIR, 'camera1', '2024-01-01');
    const cam2Dir = path.join(DEMO_DIR, 'camera2', '2024-01-01');
    fs.mkdirSync(cam1Dir, { recursive: true });
    fs.mkdirSync(cam2Dir, { recursive: true });

    return { cam1Dir, cam2Dir };
}

// Create dummy files with timestamps
function createDummyFiles(dirs, count) {
    const files = [];
    const buffer = Buffer.alloc(FILE_SIZE, 'X');

    for (let i = 0; i < count; i++) {
        const dir = i % 2 === 0 ? dirs.cam1Dir : dirs.cam2Dir;
        const filename = `video_${String(i).padStart(4, '0')}.mp4`;
        const filepath = path.join(dir, filename);

        fs.writeFileSync(filepath, buffer);

        // Set modification time to simulate age (older files first)
        const mtime = new Date(Date.now() - (count - i) * 60000); // 1 minute apart
        fs.utimesSync(filepath, mtime, mtime);

        files.push({ path: filepath, mtime, size: FILE_SIZE });
        console.log(`  Created: ${path.relative(DEMO_DIR, filepath)} (${(FILE_SIZE/1024).toFixed(0)}KB)`);
    }

    return files;
}

// Calculate total size of directory
function getDirSize(dir) {
    let total = 0;
    const items = fs.readdirSync(dir, { withFileTypes: true });

    for (const item of items) {
        const fullPath = path.join(dir, item.name);
        if (item.isDirectory()) {
            total += getDirSize(fullPath);
        } else {
            total += fs.statSync(fullPath).size;
        }
    }
    return total;
}

// Count files in directory
function countFiles(dir) {
    let count = 0;
    const items = fs.readdirSync(dir, { withFileTypes: true });

    for (const item of items) {
        const fullPath = path.join(dir, item.name);
        if (item.isDirectory()) {
            count += countFiles(fullPath);
        } else {
            count++;
        }
    }
    return count;
}

// Main demo
async function runDemo() {
    console.log('--- Step 1: Setting up demo directory ---\n');
    const dirs = setupDemoDir();

    console.log('--- Step 2: Creating dummy video files ---\n');
    const files = createDummyFiles(dirs, NUM_FILES);

    const initialSize = getDirSize(DEMO_DIR);
    const initialCount = countFiles(DEMO_DIR);

    console.log(`\n--- Step 3: Initial State ---\n`);
    console.log(`  Directory: ${DEMO_DIR}`);
    console.log(`  Total files: ${initialCount}`);
    console.log(`  Total size: ${(initialSize / 1024).toFixed(0)} KB`);
    console.log(`  Upper watermark: ${(UPPER_WATERMARK / 1024).toFixed(0)} KB`);
    console.log(`  Lower watermark: ${(LOWER_WATERMARK / 1024).toFixed(0)} KB`);
    console.log(`  Status: ${initialSize > UPPER_WATERMARK ? 'OVER THRESHOLD - needs cleanup!' : 'Under threshold'}`);

    console.log(`\n--- Step 4: ArchiveSpaceManager Configuration ---\n`);
    console.log('In a declarative pipeline, you would configure it like this:\n');
    console.log(JSON.stringify({
        modules: {
            spaceManager: {
                type: "ArchiveSpaceManager",
                props: {
                    pathToWatch: DEMO_DIR,
                    upperWaterMark: UPPER_WATERMARK,
                    lowerWaterMark: LOWER_WATERMARK,
                    samplingFreq: 10
                }
            }
        }
    }, null, 2));

    console.log(`\n--- Step 5: Simulating Space Management ---\n`);
    console.log('The ArchiveSpaceManager would:');
    console.log('1. Scan the directory and estimate size');
    console.log('2. Detect size exceeds upper watermark');
    console.log('3. Find and delete oldest files until lower watermark is reached\n');

    // Simulate what ArchiveSpaceManager does
    let currentSize = initialSize;
    let deletedCount = 0;

    // Sort files by modification time (oldest first)
    files.sort((a, b) => a.mtime - b.mtime);

    console.log('Simulating cleanup process:\n');

    for (const file of files) {
        if (currentSize <= LOWER_WATERMARK) {
            console.log(`\n  Reached lower watermark. Stopping cleanup.`);
            break;
        }

        if (fs.existsSync(file.path)) {
            console.log(`  Deleting: ${path.relative(DEMO_DIR, file.path)} (oldest file)`);
            fs.unlinkSync(file.path);
            currentSize -= file.size;
            deletedCount++;
            console.log(`    Current size: ${(currentSize / 1024).toFixed(0)} KB`);
        }
    }

    // Clean up empty directories
    for (const dir of [dirs.cam1Dir, dirs.cam2Dir]) {
        try {
            const contents = fs.readdirSync(dir);
            if (contents.length === 0) {
                fs.rmdirSync(dir);
                console.log(`  Removed empty directory: ${path.relative(DEMO_DIR, dir)}`);
            }
        } catch (e) {}
    }

    const finalSize = getDirSize(DEMO_DIR);
    const finalCount = countFiles(DEMO_DIR);

    console.log(`\n--- Step 6: Final State ---\n`);
    console.log(`  Files deleted: ${deletedCount}`);
    console.log(`  Files remaining: ${finalCount}`);
    console.log(`  Final size: ${(finalSize / 1024).toFixed(0)} KB`);
    console.log(`  Status: ${finalSize <= LOWER_WATERMARK ? 'Under lower watermark - cleanup complete!' : 'Still over threshold'}`);

    console.log(`\n--- Demo Complete ---\n`);
    console.log('The ArchiveSpaceManager module performs this same logic automatically');
    console.log('when running as part of a pipeline, continuously monitoring disk usage.');

    // Cleanup
    console.log('\nCleaning up demo directory...');
    fs.rmSync(DEMO_DIR, { recursive: true });
    console.log('Done!\n');
}

runDemo().catch(console.error);
