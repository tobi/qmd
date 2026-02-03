{
  lib,
  stdenv,
  buildNpmPackage,
  bun,
  makeWrapper,
  sqlite,
  jq,
  autoPatchelfHook,

  # GPU support
  config,
  cudaSupport ? config.cudaSupport or false,
  cudaPackages ? { },
  vulkanSupport ? stdenv.isLinux,
  vulkan-loader,
  autoAddDriverRunpath,
}:
let
  # CUDA only supported on x86_64-linux
  effectiveCudaSupport = cudaSupport && stdenv.isLinux && stdenv.hostPlatform.isx86_64;
  # Vulkan supported on all Linux
  effectiveVulkanSupport = vulkanSupport && stdenv.isLinux;
in
buildNpmPackage {
  pname = "qmd";
  version = "1.0.0";

  src = ./..;

  npmDepsHash = "sha256-vYW5aKgznbMCpU8cktnEiVcITSqCkehm+UCO0tMI6CY=";

  postPatch = ''
    cp ${./package-lock.json} package-lock.json
    # Remove win32 optional dependency
    ${jq}/bin/jq 'del(.optionalDependencies."sqlite-vec-win32-x64")' package.json > package.json.tmp
    mv package.json.tmp package.json
  '';

  makeCacheWritable = true;

  nativeBuildInputs = [
    makeWrapper
  ]
  ++ lib.optionals stdenv.isLinux [ autoPatchelfHook ]
  ++ lib.optionals effectiveCudaSupport [ autoAddDriverRunpath ];

  buildInputs = [
    sqlite
  ]
  ++ lib.optionals stdenv.isLinux [ stdenv.cc.cc.lib ]
  ++ lib.optionals effectiveCudaSupport [
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
  ]
  ++ lib.optionals effectiveVulkanSupport [ vulkan-loader ];

  autoPatchelfIgnoreMissingDeps = [
    "libc.musl-x86_64.so.1"
    "libc.musl-aarch64.so.1"
  ]
  ++ lib.optionals (!effectiveCudaSupport) [
    "libcudart.so.12"
    "libcudart.so.13"
    "libcublas.so.12"
    "libcublas.so.13"
    "libcuda.so.1"
  ]
  ++ lib.optionals effectiveCudaSupport [
    "libcudart.so.13"
    "libcublas.so.13"
    "libcuda.so.1"
  ]
  ++ lib.optionals (!effectiveVulkanSupport) [ "libvulkan.so.1" ];

  npmFlags = [ "--ignore-scripts" ];
  dontNpmBuild = true;

  installPhase =
    let
      ldLibraryPath = lib.makeLibraryPath (
        [ sqlite.out ]
        ++ lib.optionals effectiveCudaSupport [
          cudaPackages.cuda_cudart
          cudaPackages.libcublas
        ]
        ++ lib.optionals effectiveVulkanSupport [ vulkan-loader ]
      );
    in
    ''
      runHook preInstall

      mkdir -p $out/lib/qmd $out/bin
      cp -r node_modules src package.json $out/lib/qmd/

      # Patch node-llama-cpp glibc detection for NixOS
      patch -p1 -d $out/lib/qmd < ${./node-llama-cpp-detectGlibc.patch}

      makeWrapper ${bun}/bin/bun $out/bin/qmd \
        --add-flags "$out/lib/qmd/src/qmd.ts" \
        --set DYLD_LIBRARY_PATH "${sqlite.out}/lib" \
        --set LD_LIBRARY_PATH "${ldLibraryPath}"

      runHook postInstall
    '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck
    HOME=$(mktemp -d) $out/bin/qmd --help | grep -q "Usage:"
    HOME=$(mktemp -d) $out/bin/qmd status
    runHook postInstallCheck
  '';

  meta = with lib; {
    description = "Local search engine for markdown docs, knowledge bases, and meeting notes";
    homepage = "https://github.com/tobi/qmd";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "qmd";
  };
}
