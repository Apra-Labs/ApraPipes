#pragma once

enum ModelArchitectureType
{
  TRANSFORMER = 0,
  ENCODERDECODER,
  CASUALDECODER,
  PREFIXDECODER,
  BERT, // Vision Transformer
  VIT,  // Bidirectional Encoder Representations from Transformer
  AST,  // Audio Spectrogram Transformer
  VIVIT // Video Vision Transformer
};

enum UseCase
{
  TEXT_TO_TEXT = 0,
  SCENE_DESCRIPTOR,
  OCR
};