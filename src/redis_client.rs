use redis::{Client, Connection, RedisResult};

use crate::config::RedisConfig;

pub fn create_connection(host: &str, config: &RedisConfig) -> RedisResult<Connection> {
    let url = config.connection_url(host);
    let client = Client::open(url.as_str())?;
    client.get_connection()
}
