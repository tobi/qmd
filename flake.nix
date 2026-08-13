{
  description = "QMD - Quick Markdown Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    {
      homeModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.programs.qmd;
        in
        {
          options.programs.qmd = {
            enable = mkEnableOption "QMD - on-device search engine for markdown notes";

            package = mkOption {
              type = types.package;
              default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
              defaultText = literalExpression "inputs.qmd.packages.\${pkgs.stdenv.hostPlatform.system}.default";
              description = "The qmd package to use.";
            };
          };

          config = mkIf cfg.enable {
            home.packages = [ cfg.package ];
          };
        };
    } //
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        packageJson = builtins.fromJSON (builtins.readFile ./package.json);
        version = packageJson.version;

        # SQLite with loadable extension support for sqlite-vec
        sqliteWithExtensions = pkgs.sqlite.overrideAttrs (old: {
          configureFlags = (old.configureFlags or []) ++ [
            "--enable-load-extension"
          ];
        });

        nodeModulesHashes = {
          x86_64-linux = "sha256-jvq2TO0SxEV1BHyT6C32VQ916wMTM/D1nsV2rNcJQSo=";
          aarch64-darwin = "sha256-9vvR3KLmBc+4bfyWEyyM8FHWg+DfiDzUlwqUlm3NFc8=";

          # Populate these on first build for additional hosts if/when needed.
          aarch64-linux = pkgs.lib.fakeHash;
          x86_64-darwin = pkgs.lib.fakeHash;
        };

        nodeModules = pkgs.stdenvNoCC.mkDerivation {
          pname = "qmd-node-modules";
          inherit version;

          src = ./.;

          impureEnvVars = pkgs.lib.fetchers.proxyImpureEnvVars ++ [
            "GIT_PROXY_COMMAND"
            "SOCKS_SERVER"
          ];

          nativeBuildInputs = [
            pkgs.bun
          ];

          dontConfigure = true;

          buildPhase = ''
            export HOME=$(mktemp -d)

            bun install \
              --backend copyfile \
              --frozen-lockfile \
              --ignore-scripts \
              --no-progress \
              --production
          '';

          installPhase = ''
            mkdir -p $out
            cp -R node_modules $out/
          '';

          dontFixup = true;

          outputHash = nodeModulesHashes.${system};
          outputHashAlgo = "sha256";
          outputHashMode = "recursive";
        };

        qmd = pkgs.stdenv.mkDerivation {
          pname = "qmd";
          inherit version;

          src = ./.;

          nativeBuildInputs = [
            pkgs.bun
            pkgs.makeWrapper
            pkgs.nodejs
            pkgs.node-gyp
            pkgs.python3  # needed by node-gyp to compile better-sqlite3
          ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
            pkgs.darwin.cctools  # provides libtool needed by node-gyp on macOS
          ];

          buildInputs = [ pkgs.sqlite ];

          buildPhase = ''
            export HOME=$(mktemp -d)

            cp -R ${nodeModules}/node_modules ./
            chmod -R u+w node_modules

            (cd node_modules/better-sqlite3 && node-gyp rebuild --release)
          '';

          installPhase = ''
            mkdir -p $out/lib/qmd
            mkdir -p $out/bin

            cp -r node_modules $out/lib/qmd/
            cp -r src $out/lib/qmd/
            cp -r skills $out/lib/qmd/
            cp package.json $out/lib/qmd/

            # The flake wraps `bun src/cli/qmd.ts` directly, so bin/qmd never
            # runs. Mirror its pre-import env here (#723): quiet llama/ggml
            # native logs for `qmd mcp` (stdio is JSON-RPC), and disable Metal
            # residency sets on Darwin so ggml's process-static destructor
            # does not dump a stack trace after a successful query
            # (ggml-org/llama.cpp#22593). `--run` fires before bun starts, so
            # the env is in place before the native binding loads. Preserve
            # explicit user values; QMD_METAL_KEEP_RESIDENCY=1 opts back in.
            makeWrapper ${pkgs.bun}/bin/bun $out/bin/qmd \
              --add-flags "$out/lib/qmd/src/cli/qmd.ts" \
              --set DYLD_LIBRARY_PATH "${pkgs.sqlite.out}/lib" \
              --set LD_LIBRARY_PATH "${pkgs.sqlite.out}/lib${pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ":${pkgs.stdenv.cc.libc.out}/lib:${pkgs.stdenv.cc.cc.lib}/lib"}" \
              --run 'if [ "$1" = mcp ]; then export LLAMA_LOG_LEVEL="''${LLAMA_LOG_LEVEL:-error}"; export GGML_LOG_LEVEL="''${GGML_LOG_LEVEL:-error}"; export GGML_BACKEND_SILENT="''${GGML_BACKEND_SILENT:-1}"; fi; if [ "$(uname -s)" = Darwin ] && [ "''${QMD_METAL_KEEP_RESIDENCY:-}" != 1 ]; then export GGML_METAL_NO_RESIDENCY="''${GGML_METAL_NO_RESIDENCY:-1}"; fi'
          '';

          meta = with pkgs.lib; {
            description = "On-device search engine for markdown notes, meeting transcripts, and knowledge bases";
            homepage = "https://github.com/tobi/qmd";
            license = licenses.mit;
            platforms = platforms.unix;
          };
        };
      in
      {
        packages = {
          default = qmd;
          qmd = qmd;
        };

        apps.default = {
          type = "app";
          program = "${qmd}/bin/qmd";
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.bun
            sqliteWithExtensions
          ];

          shellHook = ''
            export BREW_PREFIX="''${BREW_PREFIX:-${sqliteWithExtensions.out}}"
            echo "QMD development shell"
            echo "Run: bun src/cli/qmd.ts <command>"
          '';
        };
      }
    );

}
