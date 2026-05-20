import dai
import numpy as np
import pandas as pd
import pydantic
from typing import Any, Dict, List, Optional

class Base:
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据格式统一

        根据 schema 对数据字段、类型、缺失值进行统一

        Returns:
            pd.DataFrame: 数据
        """
        # 按照 schema 确定数据的列名和顺序
        df = df.reindex(columns=self.schema.columns())
        # 按照 schema 确定数据各列的数据类型
        df = df.astype(self.schema.field_type_mapping())
        # 按照 schema 确定数据各列的默认值
        df = df.fillna(self.schema.field_default_mapping())
        return df

    def dai_write(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """数据入库

        执行数据分区和入库
        """
        # 用于前段展示
        default_docs = self.schema.default_docs()
        # 默认按照date列的年进行分区
        df[dai.DEFAULT_PARTITION_FIELD] = df["date"].dt.year.astype("int64")
        # 调用 write_bdb() 接口存放数据
        dai.DataSource.write_bdb(
            df,
            id=self.datasource_id,
            unique_together=self.unique_together,
            sort_by=self.sort_by,
            indexes=self.indexes,
            docs=default_docs,
        )

class BaseBuilder(Base):
    """数据构建"""

    def create(self) -> Optional[pd.DataFrame]:
        """历史数据构建（可选）

        增量数据可能和日期等因素有关, 相关数据统一由实例属性管理

        Raises:
            NotImplementedError: 子类实现

        Returns:
            Optional[pd.DataFrame]: 数据
        """
        raise NotImplementedError

    def build(self) -> Optional[pd.DataFrame]:
        """增量数据构建

        增量数据可能和日期等因素有关, 相关数据统一由实例属性管理

        Raises:
            NotImplementedError: 子类实现

        Returns:
            Optional[pd.DataFrame]: 数据
        """
        raise NotImplementedError


class BaseSchema(pydantic.BaseModel):
    """数据描述"""

    @staticmethod
    def is_pydantic_v2():
        return pydantic.__version__ >= "2.0.0"

    @classmethod
    def field_type_mapping_v1(cls) -> Dict[str, Any]:
        fields = cls.__fields__
        # outer_type_ 属性可能对 pydantic 版本有要求
        return {
            field: modelfield.outer_type_() if hasattr(pd, modelfield.outer_type_.__name__) else modelfield.outer_type_
            for field, modelfield in fields.items()  # type: ignore
        }

    @classmethod
    def field_type_mapping_v2(cls) -> Dict[str, Any]:
        # 要求 schema 必须写注解
        fields = cls.model_fields  # type: ignore
        schema = {}
        for field, fieldinfo in fields.items():
            value = fieldinfo.annotation
            if hasattr(pd, fieldinfo.annotation.__name__):  # type: ignore
                value = fieldinfo.annotation()  # type: ignore
            if value is np.datetime64:
                value = "datetime64[ns]"
            schema[field] = value
        return schema
        return {
            field: fieldinfo.annotation()  # type: ignore
            if hasattr(pd, fieldinfo.annotation.__name__)  # type: ignore
            else fieldinfo.annotation
            for field, fieldinfo in fields.items()
        }

    @classmethod
    def field_type_mapping(cls) -> Dict[str, Any]:
        """字段和类型的映射

        使用场景: df.astype(field_type_mapping)
        """
        if cls.is_pydantic_v2():
            return cls.field_type_mapping_v2()
        return cls.field_type_mapping_v1()

    @classmethod
    def columns_v1(cls) -> List[str]:
        """所有字段列表"""
        return list(cls.__fields__.keys())  # type: ignore

    @classmethod
    def columns_v2(cls) -> List[str]:
        """所有字段列表"""
        return list(cls.model_fields.keys())  # type: ignore

    @classmethod
    def columns(cls) -> List[str]:
        """所有字段列表"""
        if cls.is_pydantic_v2():
            return cls.columns_v2()
        return cls.columns_v1()

    @classmethod
    def field_default_mapping_v1(cls) -> Dict[str, Any]:
        fields = cls.__fields__
        return {field: modelfield.get_default() for field, modelfield in fields.items()}  # type: ignore

    @classmethod
    def field_default_mapping_v2(cls) -> Dict[str, Any]:
        fields = cls.model_fields  # type: ignore
        return {field: fieldinfo.get_default() for field, fieldinfo in fields.items()}

    @classmethod
    def field_default_mapping(cls) -> Dict[str, Any]:
        """字段和默认值的映射

        使用场景: df.fillna(field_default_mapping)
        """
        if cls.is_pydantic_v2():
            return cls.field_default_mapping_v2()
        return cls.field_default_mapping_v1()

    @classmethod
    def default_docs_v1(cls) -> Dict[str, Any]:
        fields = cls.__fields__
        return {
            "schema": {
                field: {
                    "description": modelfield.field_info.description,
                    "rank": rank * 10,
                    "group": modelfield.field_info.extra.get("group", ""),
                    "visible": modelfield.field_info.extra.get("visible", True),
                    "primary": modelfield.field_info.extra.get("primary", False),
                    "free": modelfield.field_info.extra.get("free", True),
                }
                for rank, (field, modelfield) in enumerate(fields.items())  # type: ignore
            }
        }

    @classmethod
    def default_docs_v2(cls) -> Dict[str, Any]:
        fields = cls.model_fields  # type: ignore
        return {
            "schema": {
                field: {
                    "description": fieldinfo.description,
                    "rank": rank * 10,
                    "group": (fieldinfo.json_schema_extra or {}).get("group", ""),
                    "visible": (fieldinfo.json_schema_extra or {}).get("visible", True),
                    "primary": (fieldinfo.json_schema_extra or {}).get("primary", False),
                    "free": (fieldinfo.json_schema_extra or {}).get("free", True),
                }
                for rank, (field, fieldinfo) in enumerate(fields.items())
            }
        }

    @classmethod
    def default_docs(cls) -> Dict[str, Any]:
        """表 Schema 信息

        这里只提供默认描述信息和字段顺序

        字段类型信息在 dai 中, 使用 pa.Table 生成
        """
        if cls.is_pydantic_v2():
            return cls.default_docs_v2()
        return cls.default_docs_v1()
