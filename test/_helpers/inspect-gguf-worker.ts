import { inspectGgufFile } from "../../src/llm.js";

const path = process.argv[2];
if (!path) throw new Error("A fixture path is required");
console.log(JSON.stringify(inspectGgufFile(path)));
