{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
    pros-cli-nix.url = "github:BattleCh1cken/pros-cli-nix";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    pros-cli-nix,
  }:
    utils.lib.eachDefaultSystem (system:
		let
      pkgs = import nixpkgs { inherit system; };
    in {
      devShell = pkgs.mkShell {
        packages = with pkgs; [
          pros-cli-nix.packages.${system}.default
          gcc-arm-embedded
					# clang
					clang_21
        ];
        shellHook = ''
          clear
		  echo -n Bobot go brrrr

		  export MAKEFLAGS="-j $((`nproc` - 1))"

		  touch ./.clangd
		  CLANGD=$(
		  cat << 'EOF'
CompileFlags:
  Add: [
    -std=c++23,
    "-isystem${pkgs.clang_21}/resource-root/include",

    "-isystem${pkgs.gcc-arm-embedded}/bin/../lib/gcc/arm-none-eabi/14.3.1/../../../../arm-none-eabi/include/c++/14.3.1",
    "-isystem${pkgs.gcc-arm-embedded}/bin/../lib/gcc/arm-none-eabi/14.3.1/../../../../arm-none-eabi/include/c++/14.3.1/arm-none-eabi/thumb/v7-a+simd/hard",
    "-isystem${pkgs.gcc-arm-embedded}/bin/../lib/gcc/arm-none-eabi/14.3.1/../../../../arm-none-eabi/include/c++/14.3.1/backward",
    "-isystem${pkgs.gcc-arm-embedded}/bin/../lib/gcc/arm-none-eabi/14.3.1/include",
    "-isystem${pkgs.gcc-arm-embedded}/bin/../lib/gcc/arm-none-eabi/14.3.1/include-fixed",
    "-isystem${pkgs.gcc-arm-embedded}/bin/../lib/gcc/arm-none-eabi/14.3.1/../../../../arm-none-eabi/include"
  ]
  Remove: [ -isystem*, --std=gnu++23, -mfp16-format=ieee ]
  Compiler: ${pkgs.clang_21.cc}/bin/clang
EOF
		  )
		  echo "$CLANGD" > ./.clangd

		  alias mut="pros --no-sentry --no-analytics mut --after run"
		  alias mu="pros --no-sentry --no-analytics mu"
		  alias m="pros --no-sentry --no-analytics build-compile-commands"

		  alias u="pros --no-sentry --no-analytics u"
		  alias mc="make clean && m"
		  alias t="pros --no-sentry --no-analytics t"
		  alias ut="pros --no-sentry --no-analytics ut"
        '';
      };
    });
}
