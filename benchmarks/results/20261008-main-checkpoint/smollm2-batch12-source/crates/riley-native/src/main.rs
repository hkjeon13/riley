use std::process::ExitCode;

fn main() -> ExitCode {
    let result = riley_native::parse_calibration_command(std::env::args_os().skip(1))
        .map_err(|error| error.to_string())
        .and_then(|arguments| {
            riley_native::run_calibration(&arguments).map_err(|error| error.to_string())
        });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("riley-native: {error}");
            ExitCode::FAILURE
        }
    }
}
