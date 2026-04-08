import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendRoot = path.resolve(__dirname, "..");
const distDir = path.join(frontendRoot, "dist");
const assetsDir = path.join(distDir, "assets");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";

function hasFlag(name) {
  return process.argv.includes(`--${name}`);
}

function readOption(name, fallback = "") {
  const flag = `--${name}`;
  const index = process.argv.indexOf(flag);

  if (index >= 0) {
    return process.argv[index + 1] ?? "";
  }

  const inline = process.argv.find((value) => value.startsWith(`${flag}=`));
  if (inline) {
    return inline.slice(flag.length + 1);
  }

  return fallback;
}

function trimSlashes(value) {
  return value.replace(/^\/+|\/+$/g, "");
}

function run(command, args, label) {
  console.log(`\n> ${label}`);
  console.log(`${command} ${args.join(" ")}`);

  const isWindowsCmd = process.platform === "win32" && /\.(cmd|bat)$/i.test(command);
  const result = isWindowsCmd
    ? spawnSync(process.env.ComSpec || "cmd.exe", ["/d", "/s", "/c", `${command} ${args.join(" ")}`], {
        cwd: frontendRoot,
        env: process.env,
        stdio: "inherit",
      })
    : spawnSync(command, args, {
        cwd: frontendRoot,
        env: process.env,
        stdio: "inherit",
      });

  if (result.error) {
    console.error(`\nCommand failed to start: ${result.error.message}`);
    process.exit(1);
  }

  if (typeof result.status === "number" && result.status !== 0) {
    process.exit(result.status);
  }
}

function withAwsOptions(args, profile, region) {
  const awsArgs = [];

  if (profile) {
    awsArgs.push("--profile", profile);
  }

  if (region) {
    awsArgs.push("--region", region);
  }

  return awsArgs.concat(args);
}

function assertDeployableBuild() {
  const indexHtmlPath = path.join(distDir, "index.html");

  if (!existsSync(indexHtmlPath)) {
    console.error(`Build output not found at ${indexHtmlPath}.`);
    process.exit(1);
  }

  const indexHtml = readFileSync(indexHtmlPath, "utf8");
  const entryAssetMatch = indexHtml.match(/src="(\/assets\/index-[^"]+\.js)"/);

  if (!entryAssetMatch) {
    return;
  }

  const entryAssetPath = path.join(distDir, entryAssetMatch[1].replace(/^\//, ""));
  if (!existsSync(entryAssetPath)) {
    return;
  }

  const entryAsset = readFileSync(entryAssetPath, "utf8");
  if (entryAsset.includes("replace-with-public-backend-url")) {
    console.error(
      "Build output still contains the placeholder production API URL. Update frontend/.env.production before deploying to S3.",
    );
    process.exit(1);
  }
}

const bucket = readOption("bucket", process.env.FRONTEND_S3_BUCKET ?? "").trim();
const distributionId = readOption(
  "distribution-id",
  process.env.FRONTEND_CLOUDFRONT_DISTRIBUTION_ID ?? "",
).trim();
const prefix = trimSlashes(readOption("prefix", process.env.FRONTEND_S3_PREFIX ?? "").trim());
const buildScript = readOption("build-script", process.env.FRONTEND_BUILD_SCRIPT ?? "build:production").trim();
const profile = readOption("profile", process.env.AWS_PROFILE ?? "").trim();
const region = readOption("region", process.env.AWS_REGION ?? "").trim();
const invalidationPaths = readOption("invalidation-paths", process.env.FRONTEND_INVALIDATION_PATHS ?? "/*")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const skipBuild = hasFlag("skip-build");
const skipInvalidation = hasFlag("skip-invalidation");

if (!bucket) {
  console.error("Missing S3 bucket. Pass --bucket <name> or set FRONTEND_S3_BUCKET.");
  process.exit(1);
}

if (!existsSync(path.join(frontendRoot, "package.json"))) {
  console.error(`Frontend root not found at ${frontendRoot}.`);
  process.exit(1);
}

const targetRoot = prefix ? `s3://${bucket}/${prefix}` : `s3://${bucket}`;

if (!skipBuild) {
  run(npmCommand, ["run", buildScript], `Building frontend with ${buildScript}`);
}

if (!existsSync(distDir)) {
  console.error(`Build output not found at ${distDir}.`);
  process.exit(1);
}

assertDeployableBuild();

if (existsSync(assetsDir)) {
  run(
    "aws",
    withAwsOptions(
      [
        "s3",
        "sync",
        assetsDir,
        `${targetRoot}/assets`,
        "--delete",
        "--cache-control",
        "public,max-age=31536000,immutable",
      ],
      profile,
      region,
    ),
    "Syncing hashed frontend assets to S3",
  );
}

run(
  "aws",
  withAwsOptions(
    [
      "s3",
      "sync",
      distDir,
      targetRoot,
      "--delete",
      "--exclude",
      "assets/*",
      "--exclude",
      "index.html",
      "--cache-control",
      "public,max-age=300,must-revalidate",
    ],
    profile,
    region,
  ),
  "Syncing root static files to S3",
);

run(
  "aws",
  withAwsOptions(
    [
      "s3",
      "cp",
      path.join(distDir, "index.html"),
      `${targetRoot}/index.html`,
      "--content-type",
      "text/html; charset=utf-8",
      "--cache-control",
      "no-cache, no-store, must-revalidate",
    ],
    profile,
    region,
  ),
  "Uploading index.html with no-cache headers",
);

if (distributionId && !skipInvalidation) {
  run(
    "aws",
    withAwsOptions(
      [
        "cloudfront",
        "create-invalidation",
        "--distribution-id",
        distributionId,
        "--paths",
        ...invalidationPaths,
      ],
      profile,
      region,
    ),
    "Creating CloudFront invalidation",
  );
}

console.log("\nFrontend deploy complete.");
console.log(`Target: ${targetRoot}`);

if (distributionId && !skipInvalidation) {
  console.log(`CloudFront invalidated: ${distributionId}`);
} else if (distributionId) {
  console.log("CloudFront invalidation skipped.");
} else {
  console.log("No CloudFront distribution id supplied. S3 upload only.");
}
