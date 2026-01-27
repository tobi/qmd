{
  description = "QMD - Quick Markdown Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      treefmt-nix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        treefmtEval = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true;
        };

        # SQLite with loadable extension support for sqlite-vec
        sqliteWithExtensions = pkgs.sqlite.overrideAttrs (old: {
          configureFlags = (old.configureFlags or [ ]) ++ [
            "--enable-load-extension"
          ];
        });

        # Platform-specific hashes for bun install output
        # To update: set hash to empty string, build, and copy the correct hash from error
        bunDepsHash = {
          x86_64-linux = "sha256-nkFzT3IH3fr5p5Q8FRPGtYzUkwxoM2rx95RT7nvuHd0=";
          aarch64-linux = "sha256-NekMOHckkdlcTxX4pXg2aQ+Zo3uTvBnA/RkwnjTXABg=";
          x86_64-darwin = "sha256-haJ5HP9p8hP9XcsyQXyngssn0K9W7MiWHn+ir8F2d6U=";
          aarch64-darwin = "sha256-6k2hfzTqbqqfq/9dBCb+LdP7qRRjKKyUeq43SCeQJ14=";
        };

        # FOD for bun dependencies
        bunDeps = pkgs.stdenv.mkDerivation {
          pname = "qmd-bun-deps";
          version = "1.0.0";

          src = ./.;

          impureEnvVars = pkgs.lib.fetchers.proxyImpureEnvVars ++ [
            "GIT_PROXY_COMMAND"
            "SOCKS_SERVER"
          ];

          nativeBuildInputs = [ pkgs.bun ];

          dontConfigure = true;
          dontFixup = true;

          buildPhase = ''
            runHook preBuild

            export HOME=$(mktemp -d)
            bun install --no-progress --frozen-lockfile

            runHook postBuild
          '';

          installPhase = ''
            runHook preInstall

            mkdir -p $out
            cp -R node_modules $out/

            runHook postInstall
          '';

          outputHash = bunDepsHash.${system} or (throw "Unsupported system: ${system}");
          outputHashAlgo = "sha256";
          outputHashMode = "recursive";
        };

        qmd = pkgs.stdenv.mkDerivation {
          pname = "qmd";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          buildInputs = [ pkgs.sqlite ];

          dontConfigure = true;
          dontBuild = true;

          installPhase = ''
            runHook preInstall

            mkdir -p $out/lib/qmd $out/bin

            cp -r ${bunDeps}/node_modules $out/lib/qmd/
            cp -r src $out/lib/qmd/
            cp package.json $out/lib/qmd/

            makeWrapper ${pkgs.bun}/bin/bun $out/bin/qmd \
              --add-flags "$out/lib/qmd/src/qmd.ts" \
              --set DYLD_LIBRARY_PATH "${pkgs.sqlite.out}/lib" \
              --set LD_LIBRARY_PATH "${pkgs.sqlite.out}/lib"

            runHook postInstall
          '';

          meta = with pkgs.lib; {
            description = "On-device search engine for markdown notes, meeting transcripts, and knowledge bases";
            homepage = "https://github.com/tobi/qmd";
            license = licenses.mit;
            platforms = platforms.unix;
            mainProgram = "qmd";
          };
        };
      in
      {
        formatter = treefmtEval.config.build.wrapper;

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
            echo "Run: bun src/qmd.ts <command>"
          '';
        };
      }
    );
}
