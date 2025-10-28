from sqlalchemy import Integer, String, Text
from sqlalchemy.dialects.mysql import FLOAT
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional
from all_app.db.database import Base


class Cifar100(Base):
    __tablename__ = "cifar100"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class Fashion(Base):
    __tablename__ = "fashion"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text)


class Mnist(Base):
    __tablename__ = "mnist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class AllClass(Base):
    __tablename__ = "allclass"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class Gtzan(Base):
    __tablename__ = "gtzan"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    audio: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)

class Car(Base):
    __tablename__ = "car"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    audio: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class Speech(Base):
    __tablename__ = "speech"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    audio: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class Urban(Base):
    __tablename__ = "urban"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    audio: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)


class News(Base):
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(String)
    translated_text: Mapped[str] = mapped_column(String, nullable=True)
    label: Mapped[str] = mapped_column(String)


class Code(Base):
    __tablename__ = "code"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    text: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)
