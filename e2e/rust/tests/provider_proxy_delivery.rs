// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "e2e")]

//! E2E coverage for proxy-delivered static provider credentials.
//!
//! A profile credential with `delivery: proxy` must keep both the secret and
//! the OpenShell placeholder out of the sandbox environment, and the inspected
//! proxy must replace the application's public `Authorization` value with the
//! real credential before the request reaches the upstream.

use std::io::Write;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openshell_e2e::harness::binary::openshell_cmd;
use openshell_e2e::harness::sandbox::SandboxGuard;
use tempfile::NamedTempFile;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

const PROFILE_ID: &str = "e2e-proxy-delivery";
const PROVIDER_NAME: &str = "e2e-proxy-delivery";
const TEST_HOST: &str = "host.openshell.internal";
const TOKEN_ENV: &str = "E2E_PROXY_DELIVERED_KEY";
const TEST_SECRET: &str = "e2e-proxy-delivered-secret-value";
const PUBLIC_VALUE: &str = "public-placeholder-value";
const PLACEHOLDER_PREFIX: &str = "openshell:resolve:env:";

async fn run_cli(args: &[&str]) -> (bool, String) {
    let mut command = openshell_cmd();
    command
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = command.output().await.expect("spawn openshell CLI");
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// Retry a delete until it takes effect; sandbox teardown drains
/// asynchronously and the gateway refuses to delete referenced resources.
async fn delete_until_gone(args: &[&str]) -> Result<(), String> {
    const ATTEMPTS: u32 = 40;
    let mut last_output = String::new();
    for _ in 0..ATTEMPTS {
        let (deleted, output) = run_cli(args).await;
        if deleted || output.to_lowercase().contains("not found") {
            return Ok(());
        }
        last_output = output;
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(format!(
        "'{}' still failing after {ATTEMPTS} attempts:\n{last_output}",
        args.join(" ")
    ))
}

async fn ensure_provider_resources_absent() -> Result<(), String> {
    delete_until_gone(&["provider", "delete", PROVIDER_NAME]).await?;
    delete_until_gone(&["provider", "profile", "delete", PROFILE_ID]).await
}

async fn cleanup_provider_resources() {
    if let Err(error) = ensure_provider_resources_absent().await {
        eprintln!("provider cleanup did not settle: {error}");
    }
}

fn write_provider_profile(port: u16) -> Result<NamedTempFile, String> {
    let mut file = tempfile::Builder::new()
        .suffix(".yaml")
        .tempfile()
        .map_err(|error| format!("create profile: {error}"))?;
    let profile = format!(
        r"id: {PROFILE_ID}
display_name: E2E Proxy Delivery
category: other
credentials:
  - name: api_key
    env_vars: [{TOKEN_ENV}]
    required: true
    delivery: proxy
    auth_style: bearer
endpoints:
  - host: {TEST_HOST}
    port: {port}
    protocol: rest
    access: full
binaries:
  - path: /usr/bin/python*
  - path: /usr/local/bin/python*
  - path: /sandbox/.uv/python/*/bin/python*
",
    );
    file.write_all(profile.as_bytes())
        .map_err(|error| format!("write profile: {error}"))?;
    file.flush()
        .map_err(|error| format!("flush profile: {error}"))?;
    Ok(file)
}

fn write_base_policy() -> Result<NamedTempFile, String> {
    let mut file = NamedTempFile::new().map_err(|error| format!("create policy: {error}"))?;
    file.write_all(
        br#"version: 1
filesystem_policy:
  include_workdir: true
  read_only: [/usr, /lib, /proc, /dev/urandom, /app, /etc, /var/log]
  read_write: [/sandbox, /tmp, /dev/null]
landlock:
  compatibility: best_effort
process:
  run_as_user: sandbox
  run_as_group: sandbox
"#,
    )
    .map_err(|error| format!("write policy: {error}"))?;
    file.flush()
        .map_err(|error| format!("flush policy: {error}"))?;
    Ok(file)
}

async fn import_profile(port: u16) -> Result<(), String> {
    let profile = write_provider_profile(port)?;
    let profile_path = profile
        .path()
        .to_str()
        .ok_or_else(|| "profile path is not UTF-8".to_string())?;
    let (imported, output) =
        run_cli(&["provider", "profile", "import", "--file", profile_path]).await;
    if !imported {
        return Err(format!("profile import failed:\n{output}"));
    }
    Ok(())
}

async fn create_provider(value: &str) -> (bool, String) {
    let credential = format!("{TOKEN_ENV}={value}");
    run_cli(&[
        "provider",
        "create",
        "--name",
        PROVIDER_NAME,
        "--type",
        PROFILE_ID,
        "--credential",
        &credential,
    ])
    .await
}

#[derive(Debug, Clone, Default)]
struct AuthObservation {
    authorization: Option<String>,
    saw_secret_anywhere: bool,
    saw_placeholder: bool,
}

struct HttpProbeServer {
    port: u16,
    observations: Arc<Mutex<Vec<AuthObservation>>>,
    task: JoinHandle<()>,
}

impl HttpProbeServer {
    async fn start() -> Result<Self, String> {
        let listener = TcpListener::bind(("0.0.0.0", 0))
            .await
            .map_err(|error| format!("bind HTTP probe: {error}"))?;
        let port = listener
            .local_addr()
            .map_err(|error| format!("read HTTP probe address: {error}"))?
            .port();
        let observations = Arc::new(Mutex::new(Vec::new()));
        let task_observations = Arc::clone(&observations);
        let task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let observations = Arc::clone(&task_observations);
                tokio::spawn(async move {
                    let _ = handle_http_probe(stream, observations).await;
                });
            }
        });
        Ok(Self {
            port,
            observations,
            task,
        })
    }

    async fn wait_for_observations(&self, count: usize) -> Vec<AuthObservation> {
        for _ in 0..100 {
            let observations = self.observations.lock().unwrap().clone();
            if observations.len() >= count {
                return observations;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        self.observations.lock().unwrap().clone()
    }
}

impl Drop for HttpProbeServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

fn header_value(request: &str, name: &str) -> Option<String> {
    request.lines().find_map(|line| {
        let (header, value) = line.split_once(':')?;
        header
            .trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

async fn handle_http_probe(
    mut stream: TcpStream,
    observations: Arc<Mutex<Vec<AuthObservation>>>,
) -> std::io::Result<()> {
    let mut received = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let read = tokio::time::timeout(Duration::from_secs(10), stream.read(&mut buffer)).await;
        let Ok(Ok(read)) = read else {
            break;
        };
        if read == 0 {
            break;
        }
        received.extend_from_slice(&buffer[..read]);
        if received.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }

    let request = String::from_utf8_lossy(&received).into_owned();
    let authorization = header_value(&request, "authorization");
    let observation = AuthObservation {
        authorization: authorization.clone(),
        saw_secret_anywhere: request.contains(TEST_SECRET),
        saw_placeholder: request.contains(PLACEHOLDER_PREFIX),
    };
    observations.lock().unwrap().push(observation);

    let result = match authorization.as_deref() {
        Some(value) if value == format!("Bearer {TEST_SECRET}") => "AUTH_INJECTED",
        Some(value) if value.contains(PUBLIC_VALUE) => "AUTH_PUBLIC",
        Some(_) => "AUTH_OTHER",
        None => "AUTH_MISSING",
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{result}",
        result.len()
    );
    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

/// Reports whether the credential variable is visible, then sends one request
/// through the sandbox proxy with a public `Authorization` value and prints the
/// upstream's verdict.
fn client_script(port: u16) -> String {
    format!(
        r#"
import os
import socket
import urllib.parse

host = {TEST_HOST:?}
port = {port}
token_env = {TOKEN_ENV:?}
if token_env not in os.environ:
    print("TOKEN_ABSENT")
elif os.environ[token_env].startswith({PLACEHOLDER_PREFIX:?}):
    print("TOKEN_PLACEHOLDER")
else:
    print("TOKEN_UNSAFE")
proxy_url = next(os.environ[name] for name in
                 ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")
                 if os.environ.get(name))
proxy = urllib.parse.urlparse(proxy_url)

with socket.create_connection((proxy.hostname, proxy.port or 80), timeout=10) as sock:
    target = f"{{host}}:{{port}}"
    sock.sendall(f"CONNECT {{target}} HTTP/1.1\r\nHost: {{target}}\r\n\r\n".encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk
    if not response.startswith(b"HTTP/1.1 200"):
        raise RuntimeError("CONNECT failed")
    request = (
        f"GET /v1/ping HTTP/1.1\r\nHost: {{target}}\r\n"
        f"Authorization: Bearer {PUBLIC_VALUE}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    sock.sendall(request)
    sock.settimeout(5)
    response = b""
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        response += chunk
    text = response.decode("utf-8", "replace")
    for marker in ("AUTH_INJECTED", "AUTH_PUBLIC", "AUTH_OTHER", "AUTH_MISSING"):
        if marker in text:
            print(marker)
            break
    else:
        print("AUTH_NO_RESPONSE " + text.splitlines()[0] if text else "AUTH_NO_RESPONSE")
"#
    )
}

async fn run_client_sandbox(port: u16) -> Result<String, String> {
    let policy = write_base_policy()?;
    let policy_path = policy
        .path()
        .to_str()
        .ok_or_else(|| "policy path is not UTF-8".to_string())?;
    let script = client_script(port);
    let mut sandbox = SandboxGuard::create(&[
        "--policy",
        policy_path,
        "--provider",
        PROVIDER_NAME,
        "--",
        "python3",
        "-c",
        &script,
    ])
    .await?;
    let output = sandbox.create_output.clone();
    sandbox.cleanup().await;
    Ok(output)
}

#[tokio::test]
async fn proxy_delivered_credential_stays_out_of_sandbox_and_is_injected_upstream() {
    let server = HttpProbeServer::start().await.expect("start HTTP probe");
    ensure_provider_resources_absent()
        .await
        .expect("clear stale provider resources");

    let result = async {
        import_profile(server.port).await?;

        // Create-time validation: a bearer value outside token68 is rejected
        // before it can fail on every request.
        let (created, output) = create_provider("not a token68 value").await;
        if created {
            return Err(format!(
                "provider create accepted a non-token68 bearer value:\n{output}"
            ));
        }
        if !output.contains("cannot be proxy-delivered") {
            return Err(format!("unexpected create rejection:\n{output}"));
        }
        if output.contains("not a token68 value") {
            return Err(format!("credential value echoed in error:\n{output}"));
        }

        let (created, output) = create_provider(TEST_SECRET).await;
        if !created {
            return Err(format!("provider create failed:\n{output}"));
        }

        let output = run_client_sandbox(server.port).await?;
        if !output.contains("TOKEN_ABSENT") {
            return Err(format!(
                "proxy-delivered credential must not appear in the sandbox environment:\n{output}"
            ));
        }
        if !output.contains("AUTH_INJECTED") {
            return Err(format!(
                "upstream did not receive the injected credential:\n{output}"
            ));
        }
        if output.contains(TEST_SECRET) || output.contains(PLACEHOLDER_PREFIX) {
            return Err(format!(
                "sandbox output leaked credential material:\n{output}"
            ));
        }

        let observations = server.wait_for_observations(1).await;
        if observations.len() != 1 {
            return Err(format!("expected one upstream request: {observations:?}"));
        }
        let observation = &observations[0];
        if observation.authorization.as_deref() != Some(&format!("Bearer {TEST_SECRET}")) {
            return Err(format!(
                "upstream Authorization header was not replaced: {observation:?}"
            ));
        }
        if observation.saw_placeholder {
            return Err(format!("upstream saw a placeholder: {observation:?}"));
        }
        if !observation.saw_secret_anywhere {
            return Err(format!("upstream did not see the secret: {observation:?}"));
        }
        Ok::<(), String>(())
    }
    .await;

    cleanup_provider_resources().await;
    result.expect("proxy delivery E2E");
}
