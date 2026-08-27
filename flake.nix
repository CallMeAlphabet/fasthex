{
  description = "fasthex - A very fast hex dumper";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
      in
      {
        packages.default = rustPlatform.buildRustPackage {
          pname = "fasthex";
          version = cargoToml.package.version;

          src = pkgs.lib.cleanSource ./.;

          cargoLock.lockFile = ./Cargo.lock;

          meta = with pkgs.lib; {
            description = "A very fast hex dumper";
            homepage = "https://github.com/CallMeAlphabet/fasthex";
            license = licenses.asl20;
            platforms = platforms.linux;
            mainProgram = "fasthex";
          };
        };

        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            (rustToolchain.override {
              extensions = [ "rust-src" "rust-analyzer" ];
            })
          ];
        };
      });
}
