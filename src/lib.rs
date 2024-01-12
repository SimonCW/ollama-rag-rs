#![warn(clippy::pedantic)]
#![warn(clippy::style)]

pub mod consts {
    pub const MODEL: &str = "mistral";
    pub const DEFAULT_SYSTEM_CLOWN: &str = r#"
    You are a troll LLM.

    Always reply by mocking the question asker.
    "#;
}
