# Installing ReserveD

Choose one installation per coordination scope:

| Environment | Runs as | Starts | Installation |
| --- | --- | --- | --- |
| [Linux system service](#linux-system-service) | Dedicated `reserved` account | Socket available at boot; daemon starts on first request | `/opt/reserved/venv` |
| [Linux user service](#linux-user-service) | Your account | Socket available at login; daemon starts on first request | `~/.local/share/reserved/venv` |
| [macOS user agent](#macos-user-agent) | Your account | At graphical login | `~/.local/share/reserved/venv` |

Use the Linux system service for coordination between users. Separate user
daemons have separate ledgers and cannot coordinate with each other. The macOS
recipe is a LaunchAgent for a logged-in user, not a machine-wide LaunchDaemon.

## Prerequisites and source

- Python **3.11 or later**, with `pip` and `venv` support.
- Git, or an extracted checkout of this repository.
- Linux: systemd and, for a user service, a working `systemctl --user` session.
- macOS: a graphical login session and `launchctl`.

On Debian/Ubuntu, the prerequisite packages are `python3`, `python3-venv`,
`python3-pip`, and `git`; on Fedora, `python3`, `python3-pip`, and `git`.
Check the Python version: an older distribution's default Python may be too old.
On macOS, install Python from [python.org](https://www.python.org/downloads/macos/)
or run `brew install python git` with [Homebrew](https://brew.sh/).

```sh
python3 --version
git clone https://github.com/andrewguy9/Reserved.git
cd Reserved
```

If you already have a checkout, use it instead. All commands that reference
`examples/`, `deploy/`, or `pip install .` run from the checkout root. Use a
reviewed revision and record `git rev-parse HEAD` for upgrades or rollback.

Install **from this source checkout**. The public PyPI package named
[`reserved`](https://pypi.org/project/reserved/) is a different project.
The local distribution keeps the name `reserved`; do not install or upgrade it
by name from the public index. The commands below use a
[virtual environment](https://docs.python.org/3/library/venv.html), so they work
with externally managed system Python installations without overriding them.

ReserveD has no runtime Python dependencies. CUDA, Metal, and PyTorch are not
required to run the daemon or verify installation. Workloads install their own
frameworks and drivers. Hardware is configured manually; ReserveD does not
discover devices or measure free hardware memory.

## Linux system service

This recipe requires sudo. It creates a service account and a group for clients;
the daemon itself runs without root privileges. The supplied unit files use the
exact virtual environment path installed here.
Use a machine-wide Python interpreter accessible to the service account;
a Python installed inside a private home directory may not be executable by it.

### 1. Install the package and configuration

```sh
sudo install -d -m 0755 /opt/reserved
sudo "$(command -v python3)" -m venv /opt/reserved/venv
sudo /opt/reserved/venv/bin/python -m pip install .
sudo chmod -R a+rX /opt/reserved/venv

getent group reserved >/dev/null || sudo groupadd --system reserved
getent group gpu-users >/dev/null || sudo groupadd gpu-users
id reserved >/dev/null 2>&1 || sudo useradd --system --no-create-home --gid reserved reserved
sudo -u reserved /opt/reserved/venv/bin/python -I -c 'import reserved'

sudo install -d -m 0755 /etc/reserved
sudo install -m 0644 examples/cpu.json /etc/reserved/reserved.json
```

The example reserves a 1 GiB CPU memory budget and works without a GPU. Edit
`/etc/reserved/reserved.json` to describe your machine before using real workloads;
see the [configuration reference](README.md#resource-config). The service account
must be able to read this file. When reinstalling, retain your existing config
instead of copying the example over it.

### 2. Install and enable socket activation

```sh
sudo install -m 0644 deploy/systemd/system/reserved.socket /etc/systemd/system/reserved.socket
sudo install -m 0644 deploy/systemd/system/reserved.service /etc/systemd/system/reserved.service
sudo systemctl daemon-reload
sudo systemctl enable --now reserved.socket
sudo usermod -aG gpu-users "$(id -un)"
```

Log out and back in to acquire the new group membership, then check `id -nG`.
The socket is `/var/run/reserved.sock`, owned by root with group `gpu-users` and
mode `0660`. Systemd opens it and passes the listener to the service, so the
service account does not need write access to `/var/run` or membership in
`gpu-users`. Enable the **socket** unit; the service starts when a client connects.

### 3. Configure clients and verify

In the new shell:

```sh
export PATH="/opt/reserved/venv/bin:$PATH"
export RESERVED_SOCKET=/var/run/reserved.sock
reserve-status --json
reserve install-check type=cpu,ram=1 -- true
sudo systemctl status reserved.socket reserved.service --no-pager
sudo journalctl -u reserved.service -n 50 --no-pager
```

The status response should contain `cpu`, and the wrapped command should exit
with status 0. Repeat `reserve-status` after reboot to verify boot activation.
The service can be inactive before the first request; a listening socket alone
does not prove that the daemon can start.

Add the two exports to your shell startup file (`~/.bashrc` for interactive Bash,
`~/.zshrc` for Zsh). Set `RESERVED_SOCKET` explicitly for every system service,
GUI application, or job using this shared daemon. Shell startup files do not
configure other services. This explicit path takes precedence over a user's
`XDG_RUNTIME_DIR` and prevents clients from selecting a separate user daemon.
For an application in another virtual environment, install this checkout there
too to use the SDK: `/path/to/app/venv/bin/python -m pip install /path/to/Reserved`.

## Linux user service

This recipe needs no sudo once Python and systemd user sessions are available.
Run it in a login session where `XDG_RUNTIME_DIR` is set and `systemctl --user`
can connect to the user manager. Do not create a runtime directory manually to
work around a missing user session.

### 1. Install the package and configuration

```sh
python3 -m venv "$HOME/.local/share/reserved/venv"
"$HOME/.local/share/reserved/venv/bin/python" -m pip install .
mkdir -p "$HOME/.config/reserved"
cp examples/cpu.json "$HOME/.config/reserved/reserved.json"
```

Edit the config for your hardware using the [configuration reference](README.md#resource-config).
Retain existing configuration when reinstalling. The CPU example supports the
verification commands below without a GPU.

### 2. Install and enable the units

```sh
mkdir -p "$HOME/.config/systemd/user"
cp deploy/systemd/user/reserved.socket deploy/systemd/user/reserved.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now reserved.socket
```

The socket unit uses `%t/reserved.sock`, where `%t` is your systemd runtime
directory (`$XDG_RUNTIME_DIR`, usually `/run/user/<uid>`). Its mode is `0600`.
The daemon, SDK, and CLIs all use that path automatically when
`RESERVED_SOCKET` is unset. Keep the same runtime environment in your clients;
for a job outside your login session, pass the actual socket path explicitly.

### 3. Verify and choose lifetime

```sh
export PATH="$HOME/.local/share/reserved/venv/bin:$PATH"
unset RESERVED_SOCKET
reserve-status --json
reserve install-check type=cpu,ram=1 -- true
systemctl --user status reserved.socket reserved.service --no-pager
journalctl --user -u reserved.service -n 50 --no-pager
```

Persist the PATH export in your shell startup file. Remove any old
`RESERVED_SOCKET` export that selects a different daemon.

By default, the user manager follows login-session lifetime. To make the socket
available at boot and keep the user manager after logout, enable
[lingering](https://www.freedesktop.org/software/systemd/man/latest/loginctl.html):

```sh
loginctl enable-linger "$(id -un)"
loginctl show-user "$(id -un)" -p Linger
```

Local policy may require an administrator to run the first command. Test a
status request after logging out and back in, or after reboot, as appropriate.

## macOS user agent

This recipe works with a Python 3.11+ installation on Apple Silicon or Intel
macOS. Metal workloads require compatible Apple hardware and frameworks;
the CPU-only installation check does not. The agent runs while you are logged
into the graphical session and starts on subsequent graphical logins.

### 1. Install the package and configuration

```sh
python3 -m venv "$HOME/.local/share/reserved/venv"
"$HOME/.local/share/reserved/venv/bin/python" -m pip install .
mkdir -p "$HOME/.config/reserved" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cp examples/cpu.json "$HOME/.config/reserved/reserved.json"
```

Edit the config for your hardware. For Apple Silicon, see
[the Metal example](examples/m2_mac.json) and the
[configuration reference](README.md#resource-config). The CPU and Metal
budgets are accounted independently even when they share physical memory;
choose budgets that leave room for macOS and other applications. Retain your
existing config on reinstall.

### 2. Generate and load the agent

```sh
"$HOME/.local/share/reserved/venv/bin/python" deploy/launchd/render_agent.py \
  --venv "$HOME/.local/share/reserved/venv" \
  > "$HOME/Library/LaunchAgents/com.reserved.daemon.plist"
plutil -lint "$HOME/Library/LaunchAgents/com.reserved.daemon.plist"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.reserved.daemon.plist"
```

The renderer writes absolute interpreter, config, socket, and log paths and
escapes them as XML, including home directories containing spaces or `&`.
It uses your installed virtual environment rather than assuming a Homebrew,
python.org, or `/usr/local/bin` executable location. It does not load the agent
or modify your machine itself. If the agent is already loaded, use the restart
or upgrade instructions below instead of bootstrapping it twice.

### 3. Configure clients and verify

```sh
export PATH="$HOME/.local/share/reserved/venv/bin:$PATH"
unset RESERVED_SOCKET XDG_RUNTIME_DIR
reserve-status --json
reserve install-check type=cpu,ram=1 -- true
launchctl print "gui/$(id -u)/com.reserved.daemon"
tail -n 50 "$HOME/Library/Logs/reserved.log"
```

The agent binds `/tmp/reserved-<uid>.sock`, for example
`/tmp/reserved-501.sock`. Clients select the same per-user macOS default without
an environment export. Persist the PATH export in `~/.zshrc` (or the startup file
for your shell) and remove exports left over from earlier installation recipes.
If a client deliberately sets `XDG_RUNTIME_DIR`, select the agent explicitly with
`RESERVED_SOCKET=/tmp/reserved-$(id -u).sock`. Repeat a status request after a
logout/login to verify automatic startup. A LaunchDaemon for service before
login or coordination between macOS users requires a separate deployment.

## Socket and config discovery

All clients and the standalone daemon use the same socket precedence:

1. Explicit `--socket PATH` (CLIs) or `socket_path=PATH` (SDK).
2. Nonempty `RESERVED_SOCKET`.
3. `$XDG_RUNTIME_DIR/reserved.sock`, when the variable is nonempty.
4. `/tmp/reserved-<uid>.sock` on macOS; `/var/run/reserved.sock` on Linux.

A systemd-activated daemon uses the listener supplied by systemd; its address
and permissions come from the socket unit, not `--socket` or environment values.
ReserveD expects exactly one listening Unix stream socket and verifies that the
activation environment addresses its own process.

Configuration comes from `--config`, then `RESERVED_CONFIG`, then
`~/.config/reserved/reserved.json`, then `/etc/reserved/reserved.json`.
An explicitly selected missing file is an error; it never silently falls back
to a different configuration. All supplied service definitions pass an explicit
config path.

To test in the foreground instead of installing a service:

```sh
export PATH="$HOME/.local/share/reserved/venv/bin:$PATH"
export RESERVED_SOCKET="/tmp/reserved-$(id -u)-test.sock"
reserved --config examples/cpu.json
```

Run the same two exports in a second shell, then run `reserve-status` and the
reservation check above. Stop the foreground daemon with Ctrl-C.

## Restarting, upgrading, and removing

Restarting the daemon invalidates **all active reservations** and clears the
ledger. Coordinate with running workloads before maintenance. Existing jobs
are not stopped or re-reserved automatically.

For a configuration-only change, edit the JSON and restart:

```sh
# Linux system service
sudo systemctl restart reserved.service
# Linux user service
systemctl --user restart reserved.service
# macOS
launchctl kickstart -k "gui/$(id -u)/com.reserved.daemon"
```

Rapid macOS restarts may wait for launchd's default ten-second launch throttle.

To upgrade, select the desired source revision in the checkout. Stop the service
while replacing its package, and use the same virtual environment path:

```sh
# Linux system service
sudo systemctl stop reserved.socket reserved.service
sudo /opt/reserved/venv/bin/python -m pip install --upgrade .
sudo chmod -R a+rX /opt/reserved/venv
sudo systemctl start reserved.socket

# Linux user service
systemctl --user stop reserved.socket reserved.service
"$HOME/.local/share/reserved/venv/bin/python" -m pip install --upgrade .
systemctl --user start reserved.socket

# macOS
launchctl bootout "gui/$(id -u)/com.reserved.daemon"
"$HOME/.local/share/reserved/venv/bin/python" -m pip install --upgrade .
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.reserved.daemon.plist"
```

If service definitions changed, copy/render them again using the steps above;
on Linux run `daemon-reload` before starting the socket. Repeat status and
reservation verification. To roll back, install the previous recorded source
revision using the same sequence. If the base Python interpreter was removed
or upgraded incompatibly, recreate the virtual environment at the same path
before reinstalling; virtual environments are not portable between locations.

To disable automatic startup and remove the service definition:

```sh
# Linux system service
sudo systemctl disable --now reserved.socket
sudo systemctl stop reserved.service
sudo rm /etc/systemd/system/reserved.socket /etc/systemd/system/reserved.service
sudo systemctl daemon-reload

# Linux user service
systemctl --user disable --now reserved.socket
systemctl --user stop reserved.service
rm "$HOME/.config/systemd/user/reserved.socket" "$HOME/.config/systemd/user/reserved.service"
systemctl --user daemon-reload

# macOS
launchctl bootout "gui/$(id -u)/com.reserved.daemon"
rm "$HOME/Library/LaunchAgents/com.reserved.daemon.plist"
```

After stopping the service, its dedicated virtual environment can be removed
(`/opt/reserved/venv` or `~/.local/share/reserved/venv`). Keep the config and logs
if you plan to reinstall. Remove the PATH/socket exports from your shell startup
file. Service removal does not delete the Linux account or groups. If lingering
was enabled solely for ReserveD, it can be undone with
`loginctl disable-linger "$(id -un)"`; doing so also affects other user services.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `externally-managed-environment` | Use the virtual environment's `bin/python -m pip`, not global `pip`. |
| `203/EXEC`, missing executable, or agent repeatedly exits | Verify the interpreter path in the installed definition and recreate the venv if its base Python was removed. |
| Missing or invalid config | Check the explicit config path, JSON syntax, and service-account read permissions; inspect service logs. |
| Permission denied connecting to the system socket | Check `id -nG` after a fresh login and socket group/mode with `ls -l /var/run/reserved.sock`. |
| Socket not found or wrong device list | Check `RESERVED_SOCKET`, `XDG_RUNTIME_DIR`, and the selected config. Clients of the system service should explicitly select `/var/run/reserved.sock`. |
| Socket is listening but status does not return | Inspect the service status/logs; socket activation can succeed even when package/config startup fails. |
| Another daemon is already running | Stop the foreground or old supervised instance that owns the same socket before starting this one. |
| `launchctl bootstrap` fails | Run from the graphical user's session, lint the plist, inspect logs, and check whether the label is already loaded. |
| User service disappears after logout | Decide whether to enable lingering, as described above. |

After a crash, ReserveD probes an existing socket and replaces it only when the
connection is refused. Permission errors are reported instead of treating an
inaccessible socket as stale. For systemd installations, systemd owns socket
creation and removal.
