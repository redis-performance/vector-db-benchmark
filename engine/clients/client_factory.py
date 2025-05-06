from abc import ABC
import importlib
from typing import Dict, List, Type

from engine.base_client.client import (
    BaseClient,
    BaseConfigurator,
    BaseSearcher,
    BaseUploader,
)

# Dictionary to store dynamically imported client classes
_engine_classes = {}

def _import_engine_classes(engine_name: str) -> Dict[str, Type]:
    """
    Dynamically import client classes for a specific engine.

    Args:
        engine_name: The name of the engine (e.g., 'redis', 'qdrant')

    Returns:
        Dictionary with configurator, uploader, and searcher classes
    """
    if engine_name in _engine_classes:
        return _engine_classes[engine_name]

    # Handle special case for vectorsets which uses redis prefix
    if engine_name == "vectorsets":
        module_name = f"engine.clients.vectorsets"
        class_prefix = "RedisVset"
    else:
        module_name = f"engine.clients.{engine_name}"
        # Convert first letter to uppercase for class name
        class_prefix = engine_name[0].upper() + engine_name[1:]

    try:
        module = importlib.import_module(module_name)
        configurator_class = getattr(module, f"{class_prefix}Configurator")
        uploader_class = getattr(module, f"{class_prefix}Uploader")
        searcher_class = getattr(module, f"{class_prefix}Searcher")

        _engine_classes[engine_name] = {
            "configurator": configurator_class,
            "uploader": uploader_class,
            "searcher": searcher_class
        }

        return _engine_classes[engine_name]
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import classes for engine '{engine_name}': {e}")

# Empty dictionaries that will be populated on demand
ENGINE_CONFIGURATORS = {}
ENGINE_UPLOADERS = {}
ENGINE_SEARCHERS = {}


class ClientFactory(ABC):
    def __init__(self, host):
        self.host = host
        self.engine = None

    def _create_configurator(self, experiment) -> BaseConfigurator:
        self.engine = experiment["engine"]
        engine_name = experiment["engine"]

        # Dynamically import engine classes if not already imported
        if engine_name not in _engine_classes:
            _import_engine_classes(engine_name)
            # Add to the global dictionaries for compatibility
            ENGINE_CONFIGURATORS[engine_name] = _engine_classes[engine_name]["configurator"]
            ENGINE_UPLOADERS[engine_name] = _engine_classes[engine_name]["uploader"]
            ENGINE_SEARCHERS[engine_name] = _engine_classes[engine_name]["searcher"]

        engine_configurator_class = _engine_classes[engine_name]["configurator"]
        engine_configurator = engine_configurator_class(
            self.host,
            collection_params={**experiment.get("collection_params", {})},
            connection_params={**experiment.get("connection_params", {})},
        )
        return engine_configurator

    def _create_uploader(self, experiment) -> BaseUploader:
        engine_name = experiment["engine"]
        engine_uploader_class = _engine_classes[engine_name]["uploader"]
        engine_uploader = engine_uploader_class(
            self.host,
            connection_params={**experiment.get("connection_params", {})},
            upload_params={**experiment.get("upload_params", {})},
        )
        return engine_uploader

    def _create_searchers(self, experiment) -> List[BaseSearcher]:
        engine_name = experiment["engine"]
        engine_searcher_class: Type[BaseSearcher] = _engine_classes[engine_name]["searcher"]

        engine_searchers = [
            engine_searcher_class(
                self.host,
                connection_params={**experiment.get("connection_params", {})},
                search_params=search_params,
            )
            for search_params in experiment.get("search_params", [{}])
        ]

        return engine_searchers

    def build_client(self, experiment):
        return BaseClient(
            name=experiment["name"],
            engine=experiment["engine"],
            configurator=self._create_configurator(experiment),
            uploader=self._create_uploader(experiment),
            searchers=self._create_searchers(experiment),
        )
