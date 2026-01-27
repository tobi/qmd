{
  description = "QMD - Quick Markdown Search";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      treefmt-nix,
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

        qmd = pkgs.stdenv.mkDerivation {
          pname = "qmd";
          version = "1.0.0";

          src = ./.;

          nativeBuildInputs = [
            pkgs.bun
            pkgs.makeWrapper
          ];

          buildInputs = [ pkgs.sqlite ];

          buildPhase = ''
            export HOME=$(mktemp -d)
            bun install --frozen-lockfile
          '';

          installPhase = ''
            mkdir -p $out/lib/qmd
            mkdir -p $out/bin

            cp -r node_modules $out/lib/qmd/
            cp -r src $out/lib/qmd/
            cp package.json $out/lib/qmd/

            makeWrapper ${pkgs.bun}/bin/bun $out/bin/qmd \
              --add-flags "$out/lib/qmd/src/qmd.ts" \
              --set DYLD_LIBRARY_PATH "${pkgs.sqlite.out}/lib" \
              --set LD_LIBRARY_PATH "${pkgs.sqlite.out}/lib"
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
