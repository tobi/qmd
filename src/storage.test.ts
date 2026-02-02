import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { LocalStorageBackend, S3StorageBackend, type S3StorageConfig } from "./storage";

// S3 test configuration from environment variables
// Set these to run S3 tests:
//   S3_TEST_ENDPOINT=http://localhost:9000
//   S3_TEST_BUCKET=test-bucket
//   S3_TEST_REGION=us-east-1
//   S3_TEST_ACCESS_KEY=minioadmin
//   S3_TEST_SECRET_KEY=minioadmin
const S3_TEST_CONFIG: S3StorageConfig | null = process.env.S3_TEST_ENDPOINT ? {
    endpoint: process.env.S3_TEST_ENDPOINT,
    region: process.env.S3_TEST_REGION || "us-east-1",
    credentials: process.env.S3_TEST_ACCESS_KEY && process.env.S3_TEST_SECRET_KEY ? {
        accessKeyId: process.env.S3_TEST_ACCESS_KEY,
        secretAccessKey: process.env.S3_TEST_SECRET_KEY,
    } : undefined,
} : null;

const S3_TEST_BUCKET = process.env.S3_TEST_BUCKET || "test-bucket";

describe("LocalStorageBackend", () => {
  let tempDir: string;
  let storage: LocalStorageBackend;

  beforeAll(() => {
    tempDir = mkdtempSync(join(tmpdir(), "storage-test-"));
    storage = new LocalStorageBackend();
    
    // 创建测试文件
    writeFileSync(join(tempDir, "test.txt"), "hello world");
    mkdirSync(join(tempDir, "sub"));
    writeFileSync(join(tempDir, "sub/nested.md"), "# Title\n\nContent");
  });

  afterAll(() => {
    rmSync(tempDir, { recursive: true });
  });

  test("readFile", async () => {
    const content = await storage.readFile(join(tempDir, "test.txt"));
    expect(content).toBe("hello world");
  });

  test("exists", async () => {
    expect(await storage.exists(join(tempDir, "test.txt"))).toBe(true);
    expect(await storage.exists(join(tempDir, "nonexistent.txt"))).toBe(false);
  });

  test("listFiles", async () => {
    const files = await storage.listFiles(tempDir, "**/*.txt");
    expect(files).toContain("test.txt");
  });

  test("getFileStats", async () => {
    const stats = await storage.getFileStats(join(tempDir, "test.txt"));
    expect(stats.size).toBe(11); // "hello world".length
    expect(stats.modifiedTime).toBeGreaterThan(0);
  });

  test("writeFile and deleteFile", async () => {
    const path = join(tempDir, "new.txt");
    await storage.writeFile(path, "new content");
    expect(await storage.exists(path)).toBe(true);
    expect(await storage.readFile(path)).toBe("new content");
    
    await storage.deleteFile(path);
    expect(await storage.exists(path)).toBe(false);
  });
});

// S3 tests - only run when S3_TEST_ENDPOINT is configured
// To run these tests locally, start MinIO:
//   docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
// Then create a test bucket and set environment variables:
//   export S3_TEST_ENDPOINT=http://localhost:9000
//   export S3_TEST_BUCKET=test-bucket
//   export S3_TEST_ACCESS_KEY=minioadmin
//   export S3_TEST_SECRET_KEY=minioadmin
describe.skipIf(!S3_TEST_CONFIG)("S3StorageBackend", () => {
  let storage: S3StorageBackend;
  const testPrefix = `storage-test-${Date.now()}`;

  beforeAll(() => {
    storage = new S3StorageBackend(S3_TEST_CONFIG!);
  });

  afterAll(async () => {
    // Cleanup: delete all test files
    try {
      const files = await storage.listFiles(`s3://${S3_TEST_BUCKET}/${testPrefix}`);
      for (const file of files) {
        await storage.deleteFile(`s3://${S3_TEST_BUCKET}/${testPrefix}/${file}`);
      }
    } catch {
      // Ignore cleanup errors
    }
  });

  test("writeFile and readFile", async () => {
    const path = `s3://${S3_TEST_BUCKET}/${testPrefix}/test.txt`;
    await storage.writeFile(path, "hello s3");
    const content = await storage.readFile(path);
    expect(content).toBe("hello s3");
  });

  test("exists", async () => {
    const existingPath = `s3://${S3_TEST_BUCKET}/${testPrefix}/test.txt`;
    const nonExistingPath = `s3://${S3_TEST_BUCKET}/${testPrefix}/nonexistent.txt`;
    
    expect(await storage.exists(existingPath)).toBe(true);
    expect(await storage.exists(nonExistingPath)).toBe(false);
  });

  test("listFiles", async () => {
    // Create additional files
    await storage.writeFile(`s3://${S3_TEST_BUCKET}/${testPrefix}/sub/nested.md`, "# Nested");
    await storage.writeFile(`s3://${S3_TEST_BUCKET}/${testPrefix}/another.txt`, "another");

    const files = await storage.listFiles(`s3://${S3_TEST_BUCKET}/${testPrefix}/`);
    expect(files.length).toBeGreaterThanOrEqual(2);
    expect(files).toContain("test.txt");
    expect(files).toContain("another.txt");
  });

  test("getFileStats", async () => {
    const path = `s3://${S3_TEST_BUCKET}/${testPrefix}/test.txt`;
    const stats = await storage.getFileStats(path);
    expect(stats.size).toBe(8); // "hello s3".length
    expect(stats.modifiedTime).toBeGreaterThan(0);
  });

  test("deleteFile", async () => {
    const path = `s3://${S3_TEST_BUCKET}/${testPrefix}/to-delete.txt`;
    await storage.writeFile(path, "delete me");
    expect(await storage.exists(path)).toBe(true);
    
    await storage.deleteFile(path);
    expect(await storage.exists(path)).toBe(false);
  });

  test("resolvePath", async () => {
    const path = `s3://${S3_TEST_BUCKET}/some/path`;
    const resolved = await storage.resolvePath(path);
    expect(resolved).toBe(`s3://${S3_TEST_BUCKET}/some/path`);
  });
});