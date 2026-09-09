// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-exec bearer tokens for the sandbox-local agent admission bridge.

use base64::Engine as _;
use rand::RngExt as _;
use std::collections::HashSet;
#[cfg(not(target_vendor = "apple"))]
use std::io::Write as _;
use std::os::fd::{AsRawFd as _, OwnedFd};
use std::os::unix::process::CommandExt as _;
use std::process::Command;
use std::sync::{Arc, Mutex};

pub const TOKEN_FD_ENV: &str = "OPENSHELL_AGENT_ADMISSION_TOKEN_FD";
pub const REQUIRE_CALLER_TOKEN_ENV: &str = "OPENSHELL_AGENT_ADMISSION_REQUIRE_CALLER_TOKEN";

#[derive(Clone, Debug, Default)]
pub struct AdmissionTokenRegistry(Arc<Mutex<HashSet<String>>>);

impl AdmissionTokenRegistry {
    pub fn contains(&self, token: &str) -> bool {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(token)
    }

    pub(crate) fn prepare_child(&self, command: &mut Command) -> std::io::Result<PreparedToken> {
        let mut random = [0_u8; 32];
        rand::rng().fill(&mut random);
        let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(random);

        let read_fd = prepare_token_fd(command, &token)?;

        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(token.clone());
        command.env(TOKEN_FD_ENV, read_fd.as_raw_fd().to_string());

        Ok(PreparedToken {
            read_fd,
            registration: TokenRegistration {
                registry: self.clone(),
                token,
            },
        })
    }
}

#[cfg(not(target_vendor = "apple"))]
fn prepare_token_fd(command: &mut Command, token: &str) -> std::io::Result<OwnedFd> {
    let (read_fd, write_fd) = nix::unistd::pipe2(nix::fcntl::OFlag::O_CLOEXEC)?;
    let mut writer = std::fs::File::from(write_fd);
    writer.write_all(token.as_bytes())?;
    drop(writer);

    let read_fd_raw = read_fd.as_raw_fd();
    // Keep CLOEXEC set in the multithreaded parent. The intended child
    // clears it after fork so concurrent spawns cannot inherit the token.
    #[allow(unsafe_code)]
    unsafe {
        command.pre_exec(move || clear_close_on_exec(read_fd_raw));
    }
    Ok(read_fd)
}

#[cfg(target_vendor = "apple")]
fn prepare_token_fd(command: &mut Command, token: &str) -> std::io::Result<OwnedFd> {
    use std::os::unix::fs::OpenOptionsExt as _;

    // Darwin lacks an atomic CLOEXEC pipe API on supported versions. Reserve
    // the advertised fd atomically, then replace it in the post-fork child.
    let reserved_fd: OwnedFd = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC)
        .open("/dev/null")?
        .into();
    let reserved_fd_raw = reserved_fd.as_raw_fd();
    let child_token = token.as_bytes().to_owned();
    #[allow(unsafe_code)]
    unsafe {
        command.pre_exec(move || install_child_pipe(reserved_fd_raw, &child_token));
    }
    Ok(reserved_fd)
}

#[cfg(target_vendor = "apple")]
#[allow(unsafe_code)]
fn install_child_pipe(target_fd: std::os::fd::RawFd, token: &[u8]) -> std::io::Result<()> {
    let mut pipe_fds = [-1, -1];
    unsafe {
        if libc::pipe(pipe_fds.as_mut_ptr()) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let written = libc::write(pipe_fds[1], token.as_ptr().cast(), token.len());
        if written < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if written as usize != token.len() {
            return Err(std::io::Error::from_raw_os_error(libc::EIO));
        }
        if libc::close(pipe_fds[1]) != 0
            || libc::dup2(pipe_fds[0], target_fd) < 0
            || libc::close(pipe_fds[0]) != 0
        {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

pub struct PreparedToken {
    read_fd: OwnedFd,
    registration: TokenRegistration,
}

impl PreparedToken {
    pub(crate) fn child_spawned(self) -> TokenRegistration {
        drop(self.read_fd);
        self.registration
    }
}

pub struct TokenRegistration {
    registry: AdmissionTokenRegistry,
    token: String,
}

impl Drop for TokenRegistration {
    fn drop(&mut self) {
        self.registry
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.token);
    }
}

#[cfg(not(target_vendor = "apple"))]
fn clear_close_on_exec(fd: std::os::fd::RawFd) -> std::io::Result<()> {
    use nix::fcntl::{FcntlArg, FdFlag, fcntl};

    let flags = FdFlag::from_bits_truncate(fcntl(fd, FcntlArg::F_GETFD)?);
    fcntl(fd, FcntlArg::F_SETFD(flags & !FdFlag::FD_CLOEXEC))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_receives_token_only_through_inherited_fd_and_exit_revokes_it() {
        let registry = AdmissionTokenRegistry::default();
        let mut command = Command::new("/bin/sh");
        command.stdout(std::process::Stdio::piped());
        command.arg("-c").arg(format!(
            "fd=${{{TOKEN_FD_ENV}}}; cat <&$fd; test -z \"${{TOKEN:-}}\""
        ));
        let prepared = registry.prepare_child(&mut command).expect("prepare token");
        let child = command.spawn().expect("spawn child");
        let registration = prepared.child_spawned();
        let output = child.wait_with_output().expect("wait for child");
        assert!(output.status.success());
        let token = String::from_utf8(output.stdout).expect("ASCII token");
        assert_eq!(token.len(), 43);
        assert!(registry.contains(&token));
        drop(registration);
        assert!(
            registry
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
        );
    }

    #[test]
    fn failed_spawn_revokes_registered_token() {
        let registry = AdmissionTokenRegistry::default();
        let mut command = Command::new("/definitely/not/an/executable");
        let prepared = registry.prepare_child(&mut command).expect("prepare token");
        assert!(command.spawn().is_err());
        drop(prepared);
        assert!(
            registry
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_empty()
        );
    }

    #[test]
    fn prepared_descriptor_remains_close_on_exec_in_parent() {
        let registry = AdmissionTokenRegistry::default();
        let mut command = Command::new("/bin/true");
        let prepared = registry.prepare_child(&mut command).expect("prepare token");
        let flags = nix::fcntl::FdFlag::from_bits_truncate(
            nix::fcntl::fcntl(prepared.read_fd.as_raw_fd(), nix::fcntl::FcntlArg::F_GETFD)
                .expect("read descriptor flags"),
        );
        assert!(flags.contains(nix::fcntl::FdFlag::FD_CLOEXEC));
    }
}
