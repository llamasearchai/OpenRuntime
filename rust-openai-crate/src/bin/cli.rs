//! Command-line interface for the OpenAI client

use clap::{Parser, Subcommand};
use openai_runtime_rs::{OpenAIClient, ChatRequest, Message, CompletionRequest, EmbeddingsRequest};
use std::io::{self, Write};
use tokio::io::{AsyncBufReadExt, BufReader};

#[derive(Parser)]
#[command(name = "openai-cli")]
#[command(about = "A high-performance OpenAI CLI client")]
struct Cli {
    /// OpenAI API key
    #[arg(short, long, env = "OPENAI_API_KEY")]
    api_key: String,

    /// Model to use
    #[arg(short, long, default_value = "gpt-4o-mini")]
    model: String,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat mode
    Chat {
        /// Initial system message
        #[arg(short, long)]
        system: Option<String>,
        /// Enable streaming
        #[arg(short = 'S', long)]
        stream: bool,
    },
    /// Single completion request
    Complete {
        /// Prompt text
        prompt: String,
        /// Maximum tokens
        #[arg(short, long)]
        max_tokens: Option<u32>,
        /// Temperature
        #[arg(short, long)]
        temperature: Option<f32>,
    },
    /// Generate embeddings
    Embeddings {
        /// Input text
        text: String,
        /// Embeddings model
        #[arg(short, long, default_value = "text-embedding-ada-002")]
        embeddings_model: String,
    },
    /// Show client metrics
    Metrics,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("debug")
            .init();
    }

    let client = OpenAIClient::new(&cli.api_key)?;

    match cli.command {
        Commands::Chat { system, stream } => {
            run_interactive_chat(&client, &cli.model, system, stream).await?;
        }
        Commands::Complete { prompt, max_tokens, temperature } => {
            let mut request = CompletionRequest::new(&cli.model, prompt);
            if let Some(max_tokens) = max_tokens {
                request = request.with_max_tokens(max_tokens);
            }
            if let Some(temperature) = temperature {
                request = request.with_temperature(temperature);
            }

            let response = client.completion(request).await?;
            if let Some(text) = response.text() {
                println!("{}", text);
            }
        }
        Commands::Embeddings { text, embeddings_model } => {
            let request = EmbeddingsRequest::new(embeddings_model, text);
            let response = client.embeddings(request).await?;
            
            if let Some(embedding) = response.embedding() {
                println!("Embedding vector (length: {}):", embedding.len());
                for (i, value) in embedding.iter().take(10).enumerate() {
                    println!("  [{}]: {:.6}", i, value);
                }
                if embedding.len() > 10 {
                    println!("  ... and {} more values", embedding.len() - 10);
                }
            }
        }
        Commands::Metrics => {
            let metrics = client.metrics();
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        }
    }

    Ok(())
}

async fn run_interactive_chat(
    client: &OpenAIClient,
    model: &str,
    system_message: Option<String>,
    stream: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("OpenAI Chat CLI (type 'quit' to exit)");
    println!("Model: {}", model);
    
    let mut messages = Vec::new();
    
    if let Some(system) = system_message {
        messages.push(Message::system(system));
        println!("System message set.");
    }

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        
        line.clear();
        reader.read_line(&mut line).await?;
        let input = line.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }

        if input == "clear" {
            messages.clear();
            println!("Conversation cleared.");
            continue;
        }

        messages.push(Message::user(input));

        let request = ChatRequest::new(model)
            .with_stream(stream);
        
        let mut request_with_messages = request;
        for message in &messages {
            request_with_messages = request_with_messages.add_message(message.clone());
        }

        if stream {
            print!("> ");
            io::stdout().flush()?;
            
            let mut stream = client.stream_chat_completion(request_with_messages).await?;
            let mut full_response = String::new();
            
            use futures::StreamExt;
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(response) => {
                        if let Some(choice) = response.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                print!("{}", content);
                                io::stdout().flush()?;
                                full_response.push_str(content);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("\nError: {}", e);
                        break;
                    }
                }
            }
            
            if !full_response.is_empty() {
                messages.push(Message::assistant(full_response));
            }
            println!();
        } else {
            let response = client.chat_completion(request_with_messages).await?;
            
            if let Some(content) = response.content() {
                println!("> {}", content);
                messages.push(Message::assistant(content));
            }
        }
    }

    Ok(())
}