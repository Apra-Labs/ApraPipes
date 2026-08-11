/**
 * kb-gen runner -- executed by WorkflowEngine with context primitives.
 *
 * Phases:
 *   1. setup   -- kb_setup + initial repo mapping
 *   2. scan    -- deep file/module scanning with kb_capture
 *   3. connect -- cross-cutting pattern discovery
 *   4. promote -- verify and promote INFERRED entries to CONFIRMED
 *   5. export  -- kb_export + kb_stats report
 */

export async function main(context) {
    const { agent, log, phase, args } = context;
    const { memberName, depth, autoPromote } = args;

    // ---------------------------------------------------------------
    // Phase 1: Setup + map the repo
    // ---------------------------------------------------------------
    await phase('setup');
    log('Phase 1: Setting up KB and mapping repo structure...');

    const mapResult = await agent(
        [
            'You are a codebase analyst. Your job is to set up the Knowledge Bank and map this repo.',
            '',
            'Steps:',
            '1. Run kb_setup for this repo (use the repo root as repo_path).',
            '2. Run kb_session_prime to load any existing KB entries.',
            '3. Use the Bash tool to list the top-level directory structure:',
            '   - ls -la to see all files/dirs',
            '   - Find key files: package.json, Cargo.toml, go.mod, requirements.txt, etc.',
            '   - Identify the main source directories (src/, lib/, packages/, etc.)',
            '4. Use code_map if available to get a structural overview.',
            '',
            'Return a summary of:',
            '- What language/framework this project uses',
            '- The main directories and their purposes',
            '- Key entry points and config files',
            '',
            'Keep your summary under 500 words. Focus on structure, not content.',
        ].join('\n'),
        {
            member_name: memberName,
            label: 'Repo Structure Mapping',
            model: 'cheap',
        }
    );
    log('Phase 1 complete: repo structure mapped.');

    // ---------------------------------------------------------------
    // Phase 2: Deep scan -- file-by-file KB capture
    // ---------------------------------------------------------------
    await phase('scan');
    log('Phase 2: Deep scanning files and capturing to KB...');

    const scanPrompt = depth === 'shallow'
        ? buildShallowScanPrompt()
        : buildDeepScanPrompt();

    await agent(scanPrompt, {
        member_name: memberName,
        label: 'Deep File Scan + KB Capture',
        model: 'standard',
        // Deep scans can take a while
        timeout_s: 1800,
        max_turns: 200,
    });
    log('Phase 2 complete: files scanned and captured.');

    // ---------------------------------------------------------------
    // Phase 3: Cross-cutting knowledge
    // ---------------------------------------------------------------
    await phase('connect');
    log('Phase 3: Discovering cross-cutting patterns...');

    await agent(
        [
            'You are a codebase analyst. You have already scanned individual files.',
            'Now look for CROSS-CUTTING patterns that span multiple modules.',
            '',
            'Steps:',
            '1. Run kb_query with broad terms to see what has been captured so far.',
            '2. Read 3-5 key files that connect different parts of the system',
            '   (e.g. main entry point, router, dependency injection, config).',
            '3. For each cross-cutting pattern you find, call kb_capture:',
            '   - type: "knowledge"',
            '   - Include which files/modules are involved',
            '   - Describe the pattern and WHY it exists',
            '',
            'Patterns to look for:',
            '- Error handling strategy (centralized? per-module?)',
            '- Configuration flow (env vars? config files? how do they propagate?)',
            '- Data flow between major components',
            '- Shared utilities and their consumers',
            '- Testing patterns and conventions',
            '- Build/deploy pipeline',
            '',
            'Call kb_capture for each pattern. Type should be "knowledge".',
            'If you find step-by-step procedures, use type "runbook".',
        ].join('\n'),
        {
            member_name: memberName,
            label: 'Cross-cutting Pattern Discovery',
            model: 'standard',
        }
    );
    log('Phase 3 complete: cross-cutting patterns captured.');

    // ---------------------------------------------------------------
    // Phase 4: Auto-promote INFERRED entries to CONFIRMED
    // ---------------------------------------------------------------
    if (autoPromote !== false) {
        await phase('promote');
        log('Phase 4: Verifying and promoting INFERRED entries to CONFIRMED...');

        await agent(
            [
                'You are a KB quality reviewer. Your job is to verify INFERRED entries and promote valid ones to CONFIRMED.',
                '',
                'Steps:',
                '1. Run kb_list with confidence="INFERRED" to get all INFERRED entries.',
                '2. For EACH entry, verify it is accurate:',
                '   - For context-cache entries: check that the source_files exist',
                '     (use Bash: test -f <path>). If the file exists, the entry is valid.',
                '   - For knowledge entries: briefly check that the described pattern',
                '     is real by spot-checking one of the source_files mentioned.',
                '3. For each VALID entry, call kb_promote with:',
                '   - id: the entry id',
                '   - reason: "Verified: source files exist and summary is accurate"',
                '     (reason must be at least 20 characters)',
                '4. For entries that reference files that no longer exist, call',
                '   kb_feedback to flag them instead of promoting.',
                '',
                'IMPORTANT:',
                '- Do NOT skip the verification step -- actually check files exist',
                '- Promote entries in batches, do not wait until the end',
                '- context-cache entries where the file exists are safe to promote',
                '- knowledge entries need a quick spot-check of at least one source file',
                '- Skip entries titled "test entry" or similar junk -- flag them with kb_feedback',
            ].join('\n'),
            {
                member_name: memberName,
                label: 'Auto-promote INFERRED to CONFIRMED',
                model: 'cheap',
                timeout_s: 1800,
                max_turns: 200,
            }
        );
        log('Phase 4 complete: entries promoted.');
    } else {
        log('Phase 4: Auto-promote skipped (--no-promote).');
    }

    // ---------------------------------------------------------------
    // Phase 5: Export + stats
    // ---------------------------------------------------------------
    await phase('export');
    log('Phase 5: Exporting KB and reporting stats...');

    await agent(
        [
            'Final step: export the Knowledge Bank and report coverage.',
            '',
            '1. Run kb_export to write the canonical bible (.fleet/kb-canonical.json).',
            '2. Run kb_stats to get coverage metrics.',
            '3. Run kb_list to see all entries that were captured.',
            '4. Summarize:',
            '   - Total entries captured (by type: context-cache, knowledge, runbook)',
            '   - Confidence levels (UNVERIFIED vs INFERRED)',
            '   - Any gaps or areas that need manual promotion',
            '',
            'Report the final stats clearly.',
        ].join('\n'),
        {
            member_name: memberName,
            label: 'KB Export + Stats Report',
            model: 'cheap',
        }
    );
    log('Phase 5 complete: KB exported.');
    log('KB generation workflow finished successfully.');
}

function buildShallowScanPrompt() {
    return [
        'You are a codebase analyst performing a SHALLOW scan.',
        'Focus on the most important files only.',
        '',
        'Steps:',
        '1. Identify the 10-15 most important files in this repo:',
        '   - Main entry points',
        '   - Core modules / key classes',
        '   - Configuration files',
        '   - API definitions / route handlers',
        '',
        '2. For EACH important file:',
        '   a. Read the file',
        '   b. Call kb_capture with:',
        '      - type: "context-cache"',
        '      - title: the file path',
        '      - summary: 2-3 sentence description of what this file does',
        '      - content: key exports, classes, functions, and their purposes',
        '      - symbols: main exported symbols',
        '      - source_files: [the file path]',
        '',
        '3. For any architectural patterns you notice, call kb_capture with:',
        '   - type: "knowledge"',
        '   - Describe the pattern and which files implement it',
        '',
        'Work through files systematically. Call kb_capture after analyzing each file.',
    ].join('\n');
}

function buildDeepScanPrompt() {
    return [
        'You are a codebase analyst performing a DEEP scan.',
        'Your goal: capture every meaningful file and pattern into the Knowledge Bank.',
        '',
        'Steps:',
        '1. List ALL source files (exclude node_modules, .git, dist, build, vendor).',
        '   Use: find . -type f \\( -name "*.ts" -o -name "*.js" -o -name "*.mjs" ',
        '        -o -name "*.py" -o -name "*.go" -o -name "*.rs" -o -name "*.java" ',
        '        -o -name "*.c" -o -name "*.cpp" -o -name "*.h" \\) ',
        '        -not -path "*/node_modules/*" -not -path "*/.git/*" ',
        '        -not -path "*/dist/*" -not -path "*/build/*" -not -path "*/vendor/*"',
        '',
        '2. Group files by directory/module.',
        '',
        '3. For EACH source file:',
        '   a. Read the file (you can batch-read small files)',
        '   b. Call kb_capture with:',
        '      - type: "context-cache"',
        '      - title: the file path',
        '      - summary: what this file does (2-3 sentences)',
        '      - content: key exports, classes, functions, their purposes,',
        '        important constants, and notable patterns',
        '      - symbols: exported symbols (function names, class names, constants)',
        '      - source_files: [the file path]',
        '',
        '4. After scanning each directory/module, call kb_capture with:',
        '   - type: "knowledge"',
        '   - title: "Module: <directory name>"',
        '   - Describe the module\'s responsibility, key patterns, and how it',
        '     connects to other modules',
        '',
        '5. For any step-by-step procedures you discover (build steps, deploy',
        '   procedures, test setup), call kb_capture with type "runbook".',
        '',
        'IMPORTANT:',
        '- Call kb_capture as you go, do NOT batch them all at the end',
        '- Include symbol names so kb_query can find entries by function/class name',
        '- Focus on WHAT and WHY, not line-by-line code description',
        '- Skip generated files, lockfiles, and trivial config',
        '- For test files, capture WHAT is being tested and the testing pattern,',
        '  not every individual test case',
    ].join('\n');
}
