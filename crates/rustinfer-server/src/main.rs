use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut arguments = env::args_os();
    let _program = arguments.next();
    match (arguments.next(), arguments.next()) {
        (Some(flag), None) if flag == "--version" || flag == "-V" => {
            match rustinfer_server::version_line() {
                Ok(version) => {
                    println!("{version}");
                    ExitCode::SUCCESS
                }
                Err(error) => {
                    eprintln!("rustinfer: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("usage: rustinfer --version");
            ExitCode::from(2)
        }
    }
}
