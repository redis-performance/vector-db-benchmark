use std::env;

pub struct RedisConfig {
    pub port: u16,
    pub auth: Option<String>,
    pub user: Option<String>,
    #[allow(dead_code)]
    pub cluster: bool,
}

impl RedisConfig {
    pub fn from_env() -> Self {
        Self {
            port: env::var("REDIS_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6379),
            auth: env::var("REDIS_AUTH").ok(),
            user: env::var("REDIS_USER").ok(),
            cluster: env::var("REDIS_CLUSTER")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .map(|v| v != 0)
                .unwrap_or(false),
        }
    }

    pub fn connection_url(&self, host: &str) -> String {
        let auth_part = match (&self.user, &self.auth) {
            (Some(user), Some(pass)) => format!("{}:{}@", user, pass),
            (None, Some(pass)) => format!(":{}@", pass),
            _ => String::new(),
        };
        format!("redis://{}{}:{}/", auth_part, host, self.port)
    }
}
