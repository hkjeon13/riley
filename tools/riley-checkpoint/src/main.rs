use std::collections::BTreeMap;
#[cfg(feature = "hub")]
use std::io::Write;
use std::path::Path;

use riley_checkpoint::{Plan, Result, prepare, verify};

const HELP: &str = "riley-checkpoint (offline by default)\n\
prepare --manifest PLAN.json --source REGULAR_DIRECTORY --destination NEW_DIRECTORY\n\
verify --checkpoint DIRECTORY\n\
download --manifest PLAN.json --destination NEW_DIRECTORY --allow-network [--token-env HF_TOKEN]\n\
plan --repository OWNER/MODEL [--revision main] --output NEW_PLAN.json --allow-network [--token-env HF_TOKEN]\n\
Download and plan require the separately compiled `hub` feature. Downloads accept only commits.\n\
The manifest is an explicit file/size/SHA-256 allowlist; `plan` derives it from the Hub's\n\
LFS SHA-256 and git blob ids at the commit the revision resolves to.\n\
Never serve or edit a checkpoint until this command succeeds; verify after an interruption.";

fn fail(message: impl Into<String>) -> Box<dyn std::error::Error + Send + Sync> {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse(args: &[String]) -> Result<(&str, BTreeMap<String, String>)> {
    let command = args.first().ok_or_else(|| fail(HELP))?.as_str();
    let allowed: &[&str] = match command {
        "prepare" => &["--manifest", "--source", "--destination"],
        "verify" => &["--checkpoint"],
        "download" => &[
            "--manifest",
            "--destination",
            "--allow-network",
            "--token-env",
        ],
        "plan" => &[
            "--repository",
            "--revision",
            "--output",
            "--allow-network",
            "--token-env",
        ],
        _ => return Err(fail("unknown command; use --help")),
    };
    let mut values = BTreeMap::new();
    let mut index = 1;
    while index < args.len() {
        let flag = &args[index];
        if !allowed.contains(&flag.as_str()) {
            return Err(fail(format!("unknown flag: {flag}")));
        }
        let value = if flag == "--allow-network" {
            "true".to_owned()
        } else {
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| fail(format!("missing value: {flag}")))?;
            if value.is_empty() || value.starts_with("--") {
                return Err(fail(format!("missing value: {flag}")));
            }
            value.clone()
        };
        if values.insert(flag.clone(), value).is_some() {
            return Err(fail(format!("duplicate flag: {flag}")));
        }
        index += 1;
    }
    for flag in allowed
        .iter()
        .filter(|flag| !matches!(**flag, "--token-env" | "--revision"))
    {
        if !values.contains_key(*flag) {
            return Err(fail(format!("required flag: {flag}")));
        }
    }
    Ok((command, values))
}

fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let (command, values) = parse(args)?;
    match command {
        "verify" => verify(Path::new(&values["--checkpoint"]))?,
        "prepare" => {
            let plan = Plan::read(Path::new(&values["--manifest"]))?;
            prepare(
                &plan,
                Path::new(&values["--source"]),
                Path::new(&values["--destination"]),
            )?;
        }
        "download" | "plan" => {
            #[cfg(not(feature = "hub"))]
            return Err(fail(format!(
                "{command} is unavailable: build with --features hub"
            )));
            #[cfg(feature = "hub")]
            {
                riley_checkpoint::hub::check_network_permission(true)?;
                let token = values
                    .get("--token-env")
                    .map(|name| {
                        std::env::var(name).map_err(|_| {
                            fail("requested token environment variable is unset/non-UTF-8")
                        })
                    })
                    .transpose()?;
                if command == "plan" {
                    let bytes = riley_checkpoint::hub::plan(
                        &values["--repository"],
                        values.get("--revision").map_or("main", String::as_str),
                        token,
                    )?;
                    // create_new: never overwrite an existing plan.
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&values["--output"])?
                        .write_all(&bytes)?;
                    println!("checkpoint plan: written");
                    return Ok(());
                }
                let plan = Plan::read(Path::new(&values["--manifest"]))?;
                riley_checkpoint::hub::download(
                    &plan,
                    Path::new(&values["--destination"]),
                    true,
                    token,
                )?;
            }
        }
        _ => unreachable!("parser restricts commands"),
    }
    println!("checkpoint {command}: verified");
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args: Result<Vec<String>> = std::env::args_os()
        .skip(1)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| fail("arguments must be UTF-8"))
        })
        .collect();
    match args.and_then(|args| run(&args)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("riley-checkpoint: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_options_fail_before_side_effects() {
        for args in [
            vec!["prepare", "--manifest"],
            vec!["verify", "--checkpoint", "a", "--checkpoint", "b"],
            vec!["verify", "--checkpoint", "a", "--allow-network"],
            vec!["download", "--manifest", "a", "--destination", "b"],
            vec!["verify", "--checkpoint", "--source"],
        ] {
            assert!(parse(&args.into_iter().map(String::from).collect::<Vec<_>>()).is_err());
        }
    }
}
