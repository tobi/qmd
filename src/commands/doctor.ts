import { Command, Flags } from '@oclif/core';
import { existsSync } from 'fs';
import { resolve } from 'path';
import { getPwd, getDbPath } from '../utils/paths.ts';
import { getDb } from '../database/index.ts';
import { getOllamaUrl } from '../config/constants.ts';
import { getHashesNeedingEmbedding } from '../database/db.ts';
import { runAllIntegrityChecks, autoFixIssues } from '../database/integrity.ts';

interface CheckResult {
  status: 'success' | 'warning' | 'error';
  message: string;
  details?: string[];
  fix?: string;
}

export default class DoctorCommand extends Command {
  static description = 'Check system health and diagnose issues';

  static flags = {
    fix: Flags.boolean({
      description: 'Attempt to auto-fix common issues',
      default: false,
    }),
    verbose: Flags.boolean({
      description: 'Show detailed diagnostic information',
      default: false,
    }),
    json: Flags.boolean({
      description: 'Output results as JSON',
      default: false,
    }),
    index: Flags.string({
      description: 'Index name to check',
      default: 'default',
    }),
  };

  private checks: CheckResult[] = [];

  async run(): Promise<void> {
    const { flags } = await this.parse(DoctorCommand);

    if (!flags.json) {
      this.log('');
      this.log('🔍 QMD Health Check');
      this.log('━'.repeat(42));
      this.log('');
    }

    // Run all checks
    await this.checkProjectConfiguration(flags);
    await this.checkDependencies(flags);
    await this.checkServices(flags);
    await this.checkIndexHealth(flags);
    await this.checkDataIntegrity(flags);

    // Output results
    if (flags.json) {
      this.log(JSON.stringify(this.checks, null, 2));
    } else {
      this.displaySummary();
    }
  }

  private async checkProjectConfiguration(flags: any): Promise<void> {
    const pwd = getPwd();
    const qmdDir = resolve(pwd, '.qmd');
    const dbPath = getDbPath(flags.index);

    const results: CheckResult[] = [];

    // Check for .qmd/ directory
    if (existsSync(qmdDir)) {
      results.push({
        status: 'success',
        message: '.qmd/ directory found',
        details: [`Location: ${qmdDir}`],
      });

      // Check for .gitignore
      if (existsSync(resolve(qmdDir, '.gitignore'))) {
        results.push({
          status: 'success',
          message: '.qmd/.gitignore exists',
        });
      } else {
        results.push({
          status: 'warning',
          message: '.qmd/.gitignore missing',
          fix: "Run 'qmd init --force' to create it",
        });
      }
    } else {
      results.push({
        status: 'warning',
        message: '.qmd/ directory not found',
        details: ['Using global index location'],
        fix: "Run 'qmd init' to create project-local index",
      });
    }

    // Check database
    if (existsSync(dbPath)) {
      const stats = Bun.file(dbPath).size;
      const sizeMB = (stats / (1024 * 1024)).toFixed(2);
      results.push({
        status: 'success',
        message: 'Index database exists',
        details: [`Location: ${dbPath}`, `Size: ${sizeMB} MB`],
      });

      // Get document count
      try {
        const db = getDb(flags.index);
        const count = db
          .prepare('SELECT COUNT(*) as count FROM documents WHERE active = 1')
          .get() as { count: number };
        results.push({
          status: 'success',
          message: `${count.count} documents indexed`,
        });
        db.close();
      } catch (error) {
        results.push({
          status: 'error',
          message: 'Failed to read database',
          details: [`Error: ${error}`],
        });
      }
    } else {
      results.push({
        status: 'warning',
        message: 'No index database found',
        fix: "Run 'qmd add .' to create index",
      });
    }

    this.addCategory('Project Configuration', results);
  }

  private async checkDependencies(flags: any): Promise<void> {
    const results: CheckResult[] = [];

    // Check Bun version
    try {
      const bunVersion = Bun.version;
      results.push({
        status: 'success',
        message: `Bun runtime: v${bunVersion}`,
      });
    } catch {
      results.push({
        status: 'error',
        message: 'Bun runtime not detected',
      });
    }

    // Check sqlite-vec extension
    try {
      const db = getDb(flags.index);
      db.prepare('SELECT vec_version()').get();
      results.push({
        status: 'success',
        message: 'sqlite-vec extension: loaded',
      });
      db.close();
    } catch {
      results.push({
        status: 'error',
        message: 'sqlite-vec extension: failed to load',
        fix: 'Reinstall dependencies with: bun install',
      });
    }

    this.addCategory('Dependencies', results);
  }

  private async checkServices(flags: any): Promise<void> {
    const results: CheckResult[] = [];

    // Check Ollama server
    const ollamaUrl = getOllamaUrl();
    try {
      const response = await fetch(ollamaUrl, {
        method: 'GET',
        signal: AbortSignal.timeout(2000),
      });

      if (response.ok) {
        results.push({
          status: 'success',
          message: `Ollama server: running at ${ollamaUrl}`,
        });

        // Check for models
        try {
          const tagsResponse = await fetch(`${ollamaUrl}/api/tags`);
          if (tagsResponse.ok) {
            const data = (await tagsResponse.json()) as { models: any[] };
            results.push({
              status: 'success',
              message: `${data.models.length} Ollama models available`,
            });
          }
        } catch {
          results.push({
            status: 'warning',
            message: 'Could not fetch Ollama models list',
          });
        }
      } else {
        results.push({
          status: 'warning',
          message: `Ollama server responded with status ${response.status}`,
        });
      }
    } catch (error) {
      results.push({
        status: 'error',
        message: `Ollama server not reachable at ${ollamaUrl}`,
        fix: "Start Ollama with 'ollama serve' or set OLLAMA_URL env var",
      });
    }

    this.addCategory('Services', results);
  }

  private async checkIndexHealth(flags: any): Promise<void> {
    const results: CheckResult[] = [];
    const dbPath = getDbPath(flags.index);

    if (!existsSync(dbPath)) {
      results.push({
        status: 'warning',
        message: 'No index to check',
      });
      this.addCategory('Index Health', results);
      return;
    }

    try {
      const db = getDb(flags.index);

      // Check for documents needing embeddings
      const needsEmbedding = getHashesNeedingEmbedding(db);
      if (needsEmbedding === 0) {
        results.push({
          status: 'success',
          message: 'All documents have embeddings',
        });
      } else {
        results.push({
          status: 'warning',
          message: `${needsEmbedding} documents need embeddings`,
          fix: "Run 'qmd embed' to generate embeddings",
        });
      }

      // Check WAL mode
      const walMode = db.prepare('PRAGMA journal_mode').get() as {
        journal_mode: string;
      };
      if (walMode.journal_mode.toUpperCase() === 'WAL') {
        results.push({
          status: 'success',
          message: 'WAL mode: enabled',
        });
      } else {
        results.push({
          status: 'warning',
          message: `WAL mode: disabled (${walMode.journal_mode})`,
          details: ['WAL mode improves concurrency'],
        });
      }

      // Check for FTS index
      const ftsCount = db
        .prepare("SELECT COUNT(*) as count FROM sqlite_master WHERE type='table' AND name='documents_fts'")
        .get() as { count: number };
      if (ftsCount.count > 0) {
        results.push({
          status: 'success',
          message: 'FTS5 index: created',
        });
      } else {
        results.push({
          status: 'error',
          message: 'FTS5 index: missing',
          fix: 'Database schema corrupted, recreate index',
        });
      }

      db.close();
    } catch (error) {
      results.push({
        status: 'error',
        message: 'Failed to check index health',
        details: [`Error: ${error}`],
      });
    }

    this.addCategory('Index Health', results);
  }

  private async checkDataIntegrity(flags: any): Promise<void> {
    const results: CheckResult[] = [];
    const dbPath = getDbPath(flags.index);

    if (!existsSync(dbPath)) {
      results.push({
        status: 'warning',
        message: 'No index to check',
      });
      this.addCategory('Data Integrity', results);
      return;
    }

    try {
      const db = getDb(flags.index);

      // Run all integrity checks
      const issues = runAllIntegrityChecks(db);

      if (issues.length === 0) {
        results.push({
          status: 'success',
          message: 'All data integrity checks passed',
        });
      } else {
        // Add each issue as a result
        for (const issue of issues) {
          const status = issue.severity === 'error' ? 'error' : issue.severity === 'warning' ? 'warning' : 'success';
          results.push({
            status,
            message: issue.message,
            details: issue.details,
            fix: issue.fixable ? (flags.fix ? 'Fixing...' : 'Auto-fixable with --fix flag') : issue.type === 'stale_documents' ? "Run 'qmd cleanup --older-than=90d'" : undefined,
          });
        }

        // Auto-fix if requested
        if (flags.fix) {
          const fixed = autoFixIssues(db, issues);
          if (fixed > 0) {
            results.push({
              status: 'success',
              message: `Auto-fixed ${fixed} issue(s)`,
            });
          }
        }
      }

      db.close();
    } catch (error) {
      results.push({
        status: 'error',
        message: 'Failed to check data integrity',
        details: [`Error: ${error}`],
      });
    }

    this.addCategory('Data Integrity', results);
  }

  private addCategory(name: string, results: CheckResult[]): void {
    this.checks.push(...results);

    const hasError = results.some((r) => r.status === 'error');
    const hasWarning = results.some((r) => r.status === 'warning');
    const allSuccess = results.every((r) => r.status === 'success');

    let icon = '✓';
    if (hasError) icon = '✗';
    else if (hasWarning) icon = '⚠';

    this.log(`${icon} ${name}`);
    for (const result of results) {
      const statusIcon =
        result.status === 'success' ? '✓' : result.status === 'warning' ? '⚠' : '✗';
      this.log(`  ${statusIcon} ${result.message}`);

      if (result.details) {
        for (const detail of result.details) {
          this.log(`    ${detail}`);
        }
      }

      if (result.fix) {
        this.log(`    Fix: ${result.fix}`);
      }
    }
    this.log('');
  }

  private displaySummary(): void {
    const errors = this.checks.filter((c) => c.status === 'error').length;
    const warnings = this.checks.filter((c) => c.status === 'warning').length;

    this.log('━'.repeat(42));
    if (errors === 0 && warnings === 0) {
      this.log('✓ All checks passed! QMD is ready to use.');
    } else {
      this.log(`Overall: ${warnings} warning(s), ${errors} error(s)`);
      if (warnings > 0 || errors > 0) {
        this.log("Run 'qmd doctor --fix' to attempt auto-fixes");
      }
    }
    this.log('');
  }
}
