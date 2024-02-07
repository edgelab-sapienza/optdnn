import pickle
from dataclasses import dataclass
from enum import Enum, auto, IntEnum

import tensorflow as tf
import tensorflow_model_optimization as tfmot


class ModelProblemInt(IntEnum):
    CATEGORICAL_CLASSIFICATION = 0
    BINARY_CLASSIFICATION = 1


@dataclass
class QuantizationLayerToQuantize(Enum):
    OnlyDeepLayer = auto()
    AllLayers = auto()


@dataclass
class QuantizationTechnique(Enum):
    PostTrainingQuantization = auto()
    QuantizationAwareTraining = auto()


@dataclass
class QuantizationType(Enum):
    Standard = auto()
    ForceInt8 = auto()
    WeightInt8ActivationInt16 = auto()
    AllFP16 = auto()


@dataclass
class QuantizationParameter:
    isEnabled: bool = False
    layers_to_quantize: QuantizationLayerToQuantize = QuantizationLayerToQuantize.OnlyDeepLayer
    inOutType: tf.DType = tf.float32
    quantizationTechnique: QuantizationTechnique = (
        QuantizationTechnique.PostTrainingQuantization
    )
    quantizationType: QuantizationType = QuantizationType.ForceInt8

    def __eq__(self, __value__: object) -> bool:
        if not isinstance(__value__, QuantizationParameter):
            return False
        else:
            return (
                    self.isEnabled == __value__.isEnabled
                    and self.layers_to_quantize == __value__.layers_to_quantize
                    and self.inOutType == __value__.inOutType
                    and self.quantizationTechnique == __value__.quantizationTechnique
                    and self.quantizationType == __value__.quantizationType
            )

    def quantization_has_in_out_int(self) -> bool:
        return self.layers_to_quantize == QuantizationLayerToQuantize.AllLayers and (
                    self.inOutType == tf.int8 or self.inOutType == tf.uint8)

    def get_in_out_type(self) -> tf.DType:
        return self.inOutType

    def set_in_out_type(self, t: tf.DType):
        self.inOutType = t

    def set_quantization_technique(self, technique: QuantizationTechnique):
        self.quantizationTechnique = technique

    def get_quantization_technique(self) -> QuantizationTechnique:
        return self.quantizationTechnique


class PruningScheduleType(Enum):
    Constant = auto()
    PolynomialDecay = auto()


@dataclass
class PruningPlan:
    isEnabled: bool = False
    schedule: PruningScheduleType = PruningScheduleType.PolynomialDecay
    targetSparsity: float = 0.8

    def __str__(self) -> str:
        string = ""
        if self.schedule == PruningScheduleType.PolynomialDecay:
            string += "PolynomialDecay "
        elif self.schedule == PruningScheduleType.Constant:
            string += "Constant "
        string += f"{int(self.targetSparsity * 100)}%"
        return string

    def toJSON(self) -> bytes:
        d = {}
        d["isEnabled"] = self.isEnabled
        d["schedule"] = self.schedule
        d["targetSparsity"] = self.targetSparsity
        return pickle.dumps(d)

    @staticmethod
    def fromJSON(data: bytes):
        d = pickle.loads(data)
        p = PruningPlan()
        p.isEnabled = d["isEnabled"]
        p.schedule = d["schedule"]
        p.targetSparsity = d["targetSparsity"]
        return p

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PruningPlan):
            return False
        else:
            return (
                    self.isEnabled == __value.isEnabled
                    and self.schedule == __value.schedule
                    and self.targetSparsity == __value.targetSparsity
            )

    def generate_pruning_schedule(
            self,
            number_of_steps: int,
    ) -> tfmot.sparsity.keras.PruningSchedule:
        if self.schedule is PruningScheduleType.PolynomialDecay:
            return tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.targetSparsity / 4,
                final_sparsity=self.targetSparsity,  # Percentage of cutted weights
                begin_step=0,
                end_step=number_of_steps,
            )
        else:  # Setted as default
            return tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=self.targetSparsity,
                begin_step=0,
                end_step=number_of_steps,
            )


class OptimizationParam:
    __quantization__: QuantizationParameter = QuantizationParameter()
    __pruningPlan__: PruningPlan = PruningPlan()
    __numberOfCluster__: int = 12  # 0 clustering disabled
    __clusteringIsEnabled__: bool = False

    # Quantization methods
    def toggle_quantization(self, isEnabled: bool):
        self.__quantization__.isEnabled = isEnabled

    def set_quantized_layers(self, p: QuantizationLayerToQuantize):
        self.__quantization__.layers_to_quantize = p

    def set_quantization_type(self, type: tf.DType):
        if type is tf.uint8 or type is tf.int8 or type is tf.float16:
            self.__quantization__.layers_to_quantize = type
        else:
            print(f"Type {type} not valid")

    def isQuantizationEnabled(self) -> bool:
        return self.__quantization__.isEnabled

    def quantizationHasInOutInt(self) -> bool:
        return self.__quantization__.layers_to_quantize == QuantizationLayerToQuantize.AllLayers

    # Pruning methods
    def set_pruning_schedule(self, schedule: PruningScheduleType):
        self.__pruningPlan__.schedule = schedule

    def set_pruning_target_sparsity(self, targetSparsity: float):
        self.__pruningPlan__.targetSparsity = targetSparsity

    def toggle_pruning(self, isEnabled: bool):
        self.__pruningPlan__.isEnabled = isEnabled

    def isPruningEnabled(self) -> bool:
        return self.__pruningPlan__.isEnabled

    # Clustering methods
    def isClusteringEnabled(self) -> bool:
        return self.__numberOfCluster__ > 0 and self.__clusteringIsEnabled__

    def get_number_of_cluster(self) -> int:
        return self.__numberOfCluster__

    def set_number_of_cluster(self, cluster: int):
        self.__numberOfCluster__ = cluster

    def toggle_clustering(self, isEnabled: bool):
        self.__clusteringIsEnabled__ = isEnabled

    # General method
    def generate_pruning_schedule(
            self,
            number_of_steps: int,
    ) -> tfmot.sparsity.keras.PruningSchedule:
        if self.__pruningPlan__.schedule is PruningScheduleType.PolynomialDecay:
            return tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.__pruningPlan__.targetSparsity / 4,
                final_sparsity=self.__pruningPlan__.targetSparsity,  # Percentage of cutted weights
                begin_step=0,
                end_step=number_of_steps,
            )
        else:  # Setted as default
            return tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=self.__pruningPlan__.targetSparsity,
                begin_step=0,
                end_step=number_of_steps,
            )

    def toJSON(self) -> bytes:
        d = {}
        d["pruningPlan"] = self.__pruningPlan__.toJSON()
        d["cluster"] = self.__numberOfCluster__
        d["clusteringIsEnabled"] = self.__clusteringIsEnabled__
        d["quantSettings"] = self.__quantization__

        return pickle.dumps(d)

    @staticmethod
    def fromJSON(data: bytes):
        d = pickle.loads(data)
        optParam = OptimizationParam()
        optParam.__pruningPlan__ = PruningPlan.fromJSON(d["pruningPlan"])
        optParam.__numberOfCluster__ = d["cluster"]
        optParam.__clusteringIsEnabled__ = d["clusteringIsEnabled"]
        optParam.__quantization__ = d["quantSettings"]
        return optParam

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, OptimizationParam):
            return False
        else:
            return (
                    self.__pruningPlan__ == __value.__pruningPlan__
                    and self.__numberOfCluster__ == __value.__numberOfCluster__
                    and self.__clusteringIsEnabled__ == __value.__clusteringIsEnabled__
                    and self.__quantization__ == __value.__quantization__
            )
