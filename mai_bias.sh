#!/usr/bin/env bash
set -e

# ==============================================
# MAI-BIAS INSTALLER & RUNNER (LOGGED FOREGROUND)
# ==============================================

# ----------- ANSI Colors -----------
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
GRAY="\033[90m"
RESET="\033[37m"

# ----------- ASCII Logos -----------
function draw_mammoth() {
echo -e "$GRAY"
cat << "EOF"
          _.-- ,.--.
        .'   .'      /
        | @       |'..--------._
       /      \._/              '.
      /  .-.-                     \
     (  /    \                     \
     \\      '.                  | #
      \\       \   -.           /
       :\       |    )._____.'   \
        "       |   /  \  |  \    )
                |   |./'  :__ \.-'
                '--'
EOF
echo -e "$RESET"
}

function draw_mammoth_front() {
echo -e "$GRAY"
cat << "EOF"
              @@@@@@@@@@@   @@@@@   @@@@@@@@@@@
         @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@
     @@@@@@@@@@@@@@  @@@@@@@@@ @@@@@@@@@@ @@@@@@@@@@@@@@
      @@@@@@@@@@@@   %@@@@@@@@ @@@@@@@@@   @@@@@@@@@@@@
       @@@@@@@@@@@    @@@@@@@@ @@@@@@@@     @@@@@@@@@@
        @@@@@@@@@      @@@@@@@ @@@@@@@      @@@@@@@@@
          @@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@@@
            @@@@        @@@@@@ @@@@@@@       @@@@@
              @@@      @@@@@@@ @@@@@@@@     @@@
                 @@@@@@@@@@@@@ @@@@@@@@@@@@@
                        @@@@@@ @@@@@@
                      @@@@ @@@ @@@@@@@
                     @@@@  @@@ @@@  @@@@
                  @@@@      @@ @@@     @@@@
                            @@ @@
                            @@ @@
                            @@ @@
                             @ @
EOF
echo -e "$RESET"
}

# ----------- Step Helpers -----------
function step() { echo -e "\n${CYAN}========= Step $1: $2 =========${RESET}"; }
function ok() { echo -e "${GREEN}OK${RESET}"; }
function fail() { echo -e "${RED}FAILED${RESET}"; exit 1; }

# ----------- Intro -----------
echo -e "\n${CYAN}================ MAI-BIAS TOOLKIT ================${RESET}"
draw_mammoth
echo -e "${RESET}Installing and launching ${CYAN}MAI-BIAS${RESET}..."

# ----------- Step 1: Python 3.9+ -----------
step "1/3" "Python 3.9+ Environment"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON=$(command -v python)
else
    echo -e "${RED}Python not found.${RESET} Please install Python 3.9+."
    fail
fi

if ! $PYTHON -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
    echo -e "${RED}Python version < 3.9 detected.${RESET}"
    fail
fi
echo -n "Found $($PYTHON --version) ... "
ok

# ----------- Step 2: Virtual Environment -----------
APP_DIR="$(realpath "$(pwd)")"
VENV_DIR="$APP_DIR/venv"
echo -n "Virtual environment... "
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv "$VENV_DIR" || fail
fi
ok

# ----------- Step 3: Activate -----------
ACTIVATE_PATH=""
if [ -f "$VENV_DIR/bin/activate" ]; then
    ACTIVATE_PATH="$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    ACTIVATE_PATH="$VENV_DIR/Scripts/activate"
else
    echo -e "${RED}Activation script not found.${RESET}"
    fail
fi

# Activate environment
source "$ACTIVATE_PATH"
ok

# ----------- Step 4: Install -----------
step "2/3" "Installing MAI-BIAS"
echo -n "Upgrading pip... "
pip install --upgrade pip >/dev/null 2>&1 && ok || fail

echo -n "Installing mai_bias... "
pip install --upgrade mai_bias && ok || fail

# ----------- Step 5: Launch Foreground (Logged) -----------
step "3/3" "Running MAI-BIAS (with log)"
draw_mammoth_front
mkdir -p "$APP_DIR/.cache"
LOG_FILE="$APP_DIR/.cache/log.txt"
echo -e "${YELLOW}Starting MAI-BIAS — output is also being logged to $LOG_FILE${RESET}"
echo -e "${CYAN}Press Ctrl+C to stop.${RESET}\n"

# Run foreground process with live output + logging
python -m mai_bias.app 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}MAI-BIAS exited successfully.${RESET}"
else
    echo -e "\n${RED}MAI-BIAS exited with code $EXIT_CODE.${RESET}"
    echo -e "See log file: ${YELLOW}$LOG_FILE${RESET}"
fi
