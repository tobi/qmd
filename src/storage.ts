import { stat, readFile, writeFile, unlink } from "node:fs/promises";
import { Glob } from "bun";
import {
  S3Client,
  GetObjectCommand,
  ListObjectsV2Command,
  HeadObjectCommand,
  PutObjectCommand,
  DeleteObjectCommand,
} from "@aws-sdk/client-s3";
import { loadConfig } from "./collections";

export interface StorageBackend {
  readFile(path: string, encoding?: BufferEncoding): Promise<string>;
  writeFile(
    path: string,
    content: string,
    encoding?: BufferEncoding,
  ): Promise<void>;
  deleteFile(path: string): Promise<void>;
  listFiles(basePath: string, pattern?: string): Promise<string[]>;
  exists(path: string): Promise<boolean>;
  getFileStats(
    path: string,
  ): Promise<{ size: number; modifiedTime: number; createdTime: number }>;
  resolvePath(path: string): Promise<string>;
}

export class LocalStorageBackend implements StorageBackend {
  async readFile(
    path: string,
    encoding: BufferEncoding = "utf-8",
  ): Promise<string> {
    return await readFile(path, { encoding });
  }
  async writeFile(
    path: string,
    content: string,
    encoding: BufferEncoding = "utf-8",
  ): Promise<void> {
    await writeFile(path, content, { encoding });
  }
  async deleteFile(path: string): Promise<void> {
    await unlink(path);
  }
  async listFiles(
    basePath: string,
    pattern: string = "**/*",
  ): Promise<string[]> {
    const glob = new Glob(pattern);
    let files: string[] = [];
    for await (const file of glob.scan({
      cwd: basePath,
      onlyFiles: true,
      followSymlinks: true,
    })) {
      files.push(file);
    }
    return files;
  }
  async exists(path: string): Promise<boolean> {
    try {
      await stat(path);
      return true;
    } catch {
      return false;
    }
  }
  async getFileStats(
    path: string,
  ): Promise<{ size: number; modifiedTime: number; createdTime: number }> {
    const stats = await stat(path);
    return {
      size: stats.size,
      modifiedTime: stats.mtime.getTime(),
      createdTime: stats.birthtime.getTime(),
    };
  }
  async resolvePath(path: string): Promise<string> {
    return path;
  }
}

export interface S3StorageConfig {
  region?: string;
  endpoint?: string;
  credentials?: {
    accessKeyId: string;
    secretAccessKey: string;
  };
}

export class S3StorageBackend implements StorageBackend {
  private s3Client: S3Client;

  constructor(config?: S3StorageConfig) {
    // Use provided config or load from config file
    const fileConfig = loadConfig().s3;
    const s3Config = config ?? fileConfig;
    if (!s3Config) {
      throw new Error("S3 config not found");
    }
    this.s3Client = new S3Client({
      region: s3Config.region,
      endpoint: s3Config.endpoint,
      // Use explicit credentials if provided, otherwise AWS SDK uses env vars or ~/.aws/credentials
      credentials: config?.credentials,
      forcePathStyle: true, // Required for MinIO and some S3-compatible services
    });
  }

  async readFile(
    path: string,
    encoding: BufferEncoding = "utf-8",
  ): Promise<string> {
    const { bucketName, key } = this.resolveS3Path(path);
    const command = new GetObjectCommand({
      Bucket: bucketName,
      Key: key,
    });
    const response = await this.s3Client.send(command);
    if (!response.Body) {
      throw new Error(`Failed to read S3 object: ${path}`);
    }
    return response.Body.transformToString(encoding);
  }
  async writeFile(
    path: string,
    content: string,
    encoding: BufferEncoding = "utf-8",
  ): Promise<void> {
    const { bucketName, key } = this.resolveS3Path(path);
    await this.s3Client.send(
      new PutObjectCommand({
        Bucket: bucketName,
        Key: key,
        Body: Buffer.from(content, encoding),
      }),
    );
  }
  async deleteFile(path: string): Promise<void> {
    const { bucketName, key } = this.resolveS3Path(path);
    await this.s3Client.send(
      new DeleteObjectCommand({
        Bucket: bucketName,
        Key: key,
      }),
    );
  }
  async listFiles(
    basePath: string,
    pattern: string = "**/*",
  ): Promise<string[]> {
    const { bucketName, key: prefix } = this.resolveS3Path(basePath);
    const files: string[] = [];
    let continuationToken: string | undefined;

    // Handle pagination - S3 returns max 1000 objects per request
    do {
      const command = new ListObjectsV2Command({
        Bucket: bucketName,
        Prefix: prefix,
        ContinuationToken: continuationToken,
      });
      const response = await this.s3Client.send(command);

      if (response.Contents) {
        for (const item of response.Contents) {
          if (item.Key) {
            // Return relative path (remove prefix)
            const relativePath = prefix
              ? item.Key.slice(prefix.length).replace(/^\//, "")
              : item.Key;
            if (relativePath) {
              files.push(relativePath);
            }
          }
        }
      }

      continuationToken = response.IsTruncated
        ? response.NextContinuationToken
        : undefined;
    } while (continuationToken);

    // Note: S3 doesn't support glob patterns natively, pattern parameter is ignored
    // TODO: Add client-side glob filtering with minimatch if needed
    return files;
  }
  async exists(path: string): Promise<boolean> {
    try {
      const { bucketName, key } = this.resolveS3Path(path);
      const command = new HeadObjectCommand({
        Bucket: bucketName,
        Key: key,
      });
      await this.s3Client.send(command);
      return true;
    } catch (error: any) {
      // S3 throws NotFound or 404 when object doesn't exist
      if (
        error.name === "NotFound" ||
        error.$metadata?.httpStatusCode === 404
      ) {
        return false;
      }
      throw error;
    }
  }
  async getFileStats(
    path: string,
  ): Promise<{ size: number; modifiedTime: number; createdTime: number }> {
    const { bucketName, key } = this.resolveS3Path(path);
    const command = new HeadObjectCommand({
      Bucket: bucketName,
      Key: key,
    });
    const response = await this.s3Client.send(command);
    return {
      size: response.ContentLength ?? 0,
      modifiedTime: response.LastModified?.getTime() ?? 0,
      createdTime: response.LastModified?.getTime() ?? 0,
    };
  }
  async resolvePath(path: string): Promise<string> {
    const { bucketName, key } = this.resolveS3Path(path);
    return `s3://${bucketName}/${key}`;
  }
  private resolveS3Path(path: string): { bucketName: string; key: string } {
    // Handle s3://bucket/key or s3://bucket/ or s3://bucket
    const fullPath = path.replace(/^s3:\/\//, "");
    const parts = fullPath.split("/");
    const bucketName = parts[0];
    const key = parts.slice(1).join("/");
    if (!bucketName) {
      throw new Error(`Invalid S3 path (missing bucket): ${path}`);
    }
    return { bucketName, key };
  }
}

export function createStorageBackend(path: string): StorageBackend {
  if (path.startsWith("s3://")) {
    return new S3StorageBackend();
  }
  return new LocalStorageBackend();
}
