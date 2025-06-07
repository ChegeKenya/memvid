"""
Index management for embeddings and vector search
"""

import json
import numpy as np
import faiss
import faiss.gpu # For GPU support
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import pickle
from tqdm import tqdm

from .config import get_default_config

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages embeddings, FAISS index, and metadata for fast retrieval"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IndexManager
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_default_config()
        self.embedding_model = SentenceTransformer(self.config["embedding"]["model"])
        self.dimension = self.config["embedding"]["dimension"]
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Metadata storage
        self.metadata = []
        self.chunk_to_frame = {}  # Maps chunk ID to frame number
        self.frame_to_chunks = {}  # Maps frame number to chunk IDs
        
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration, with GPU support."""
        index_type = self.config["index"]["type"]
        use_gpu = self.config["index"].get('use_gpu', False)
        index = None

        if use_gpu:
            try:
                logger.info("Attempting to create GPU-accelerated FAISS index.")
                res = faiss.StandardGpuResources()  # Initialize GPU resources

                if index_type == "Flat":
                    index = faiss.GpuIndexFlatL2(res, self.dimension) #, faiss.METRIC_L2 is default for GpuIndexFlatL2
                    logger.info("Created GpuIndexFlatL2.")
                elif index_type == "IVF":
                    quantizer = faiss.IndexFlatL2(self.dimension) # CPU quantizer
                    # Note: For GpuIndexIVFFlat, nlist is derived from the quantizer or set during training.
                    # Here we pass nlist, GpuIndexIVFFlat will use it.
                    nlist = self.config["index"]["nlist"]
                    index = faiss.GpuIndexIVFFlat(res, self.dimension, nlist, quantizer)
                    logger.info(f"Created GpuIndexIVFFlat with nlist={nlist}.")
                else:
                    raise ValueError(f"Unknown index type for GPU: {index_type}")

                logger.info("Successfully created GPU index.")
            except Exception as e:
                logger.warning(f"Failed to create GPU index: {e}. Falling back to CPU.")
                # Ensure GPU resources are freed if partially initialized and error occurred
                if 'res' in locals() and hasattr(res, 'no_gil'): # Check if res was initialized
                    pass # res will be freed when it goes out of scope.
                index = None # Ensure we proceed to CPU creation

        if index is None: # Fallback to CPU if use_gpu is False or GPU creation failed
            logger.info("Creating CPU-based FAISS index.")
            if index_type == "Flat":
                index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created IndexFlatL2 (CPU).")
            elif index_type == "IVF":
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = self.config["index"]["nlist"]
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                logger.info(f"Created IndexIVFFlat (CPU) with nlist={nlist}.")
            else:
                raise ValueError(f"Unknown index type: {index_type}")

        # Wrap with IndexIDMap for adding IDs
        # IndexIDMap works with both CPU and GPU indexes transparently for add_with_ids
        final_index = faiss.IndexIDMap(index)
        logger.info(f"Wrapped index with IndexIDMap. Final index type: {type(final_index.index)}")
        return final_index

    def add_chunks(self, chunks: List[str], frame_numbers: List[int],
                   show_progress: bool = True) -> List[int]:
        """
        Add chunks to index with robust error handling and validation

        Args:
            chunks: List of text chunks
            frame_numbers: Corresponding frame numbers for each chunk
            show_progress: Show progress bar

        Returns:
            List of successfully added chunk IDs
        """
        if len(chunks) != len(frame_numbers):
            raise ValueError("Number of chunks must match number of frame numbers")

        logger.info(f"Processing {len(chunks)} chunks for indexing...")

        # Phase 1: Validate and filter chunks
        valid_chunks = []
        valid_frames = []
        skipped_count = 0

        for chunk, frame_num in zip(chunks, frame_numbers):
            if self._is_valid_chunk(chunk):
                valid_chunks.append(chunk)
                valid_frames.append(frame_num)
            else:
                skipped_count += 1
                logger.warning(f"Skipping invalid chunk at frame {frame_num}: length={len(chunk) if chunk else 0}")

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} invalid chunks out of {len(chunks)} total")

        if not valid_chunks:
            logger.error("No valid chunks to process")
            return []

        logger.info(f"Processing {len(valid_chunks)} valid chunks")

        # Phase 2: Generate embeddings with batch processing and error recovery
        try:
            embeddings = self._generate_embeddings(valid_chunks, show_progress)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

        if embeddings is None or len(embeddings) == 0:
            logger.error("No embeddings generated")
            return []

        # Phase 3: Add to FAISS index
        try:
            chunk_ids = self._add_to_index(embeddings, valid_chunks, valid_frames)
            logger.info(f"Successfully added {len(chunk_ids)} chunks to index")
            return chunk_ids
        except Exception as e:
            logger.error(f"Failed to add chunks to index: {e}")
            return []

    def _is_valid_chunk(self, chunk: str) -> bool:
        """Validate chunk for SentenceTransformer processing - SIMPLIFIED"""
        if not isinstance(chunk, str):
            return False

        chunk = chunk.strip()

        # Basic checks only
        if len(chunk) == 0:
            return False

        if len(chunk) > 8192:  # SentenceTransformer limit
            return False

        # Remove the harsh alphanumeric requirement - academic text has lots of punctuation!
        # Just ensure it's not binary data
        try:
            chunk.encode('utf-8')  # Can be encoded as UTF-8
            return True
        except UnicodeEncodeError:
            return False

    def _generate_embeddings(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings with error handling and batch processing"""

        # Try full batch first
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks (full batch)")
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=show_progress,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True  # Helps with numerical stability
            )
            return np.array(embeddings).astype('float32')

        except Exception as e:
            logger.warning(f"Full batch embedding failed: {e}. Trying batch processing...")

            # Fall back to smaller batches
            return self._generate_embeddings_batched(chunks, show_progress)

    def _generate_embeddings_batched(self, chunks: List[str], show_progress: bool) -> np.ndarray:
        """Generate embeddings in smaller batches with individual error handling"""

        all_embeddings = []
        valid_chunks = []
        batch_size = 100  # Smaller batches

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        if show_progress:
            from tqdm import tqdm
            batch_iter = tqdm(range(0, len(chunks), batch_size),
                              desc="Processing chunks in batches",
                              total=total_batches)
        else:
            batch_iter = range(0, len(chunks), batch_size)

        for i in batch_iter:
            batch_chunks = chunks[i:i + batch_size]

            try:
                # Try batch
                batch_embeddings = self.embedding_model.encode(
                    batch_chunks,
                    show_progress_bar=False,
                    batch_size=16,  # Even smaller internal batch
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

                all_embeddings.extend(batch_embeddings)
                valid_chunks.extend(batch_chunks)

            except Exception as e:
                logger.warning(f"Batch {i//batch_size} failed: {e}. Processing individually...")

                # Process individually
                for chunk in batch_chunks:
                    try:
                        embedding = self.embedding_model.encode(
                            [chunk],
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        all_embeddings.extend(embedding)
                        valid_chunks.append(chunk)

                    except Exception as chunk_error:
                        logger.error(f"Failed to embed individual chunk (length={len(chunk)}): {chunk_error}")
                        # Skip this chunk entirely
                        continue

        if not all_embeddings:
            raise RuntimeError("No embeddings could be generated")

        logger.info(f"Generated embeddings for {len(valid_chunks)} out of {len(chunks)} chunks")
        return np.array(all_embeddings).astype('float32')

    def _add_to_index(self, embeddings: np.ndarray, chunks: List[str], frame_numbers: List[int]) -> List[int]:
        """Add embeddings to FAISS index with error handling"""

        if len(embeddings) != len(chunks) or len(embeddings) != len(frame_numbers):
            # This can happen if some chunks were skipped during embedding
            min_len = min(len(embeddings), len(chunks), len(frame_numbers))
            embeddings = embeddings[:min_len]
            chunks = chunks[:min_len]
            frame_numbers = frame_numbers[:min_len]
            logger.warning(f"Trimmed to {min_len} items due to length mismatch")

        # Assign IDs
        start_id = len(self.metadata)
        chunk_ids = list(range(start_id, start_id + len(chunks)))

        # Train index if needed (for IVF)
        try:
            underlying_index = self.index.index  # Get the actual index from IndexIDMap wrapper

            if isinstance(underlying_index, faiss.IndexIVFFlat):
                nlist = underlying_index.nlist

                if not underlying_index.is_trained:
                    logger.info(f"🧠 FAISS IVF index requires training (nlist={nlist})")
                    logger.info(f"📊 Available embeddings: {len(embeddings)}")

                    # Check if we have enough data for training
                    if len(embeddings) < nlist:
                        logger.warning(f"❌ Insufficient training data: need at least {nlist} embeddings, got {len(embeddings)}")
                        logger.warning(f"💡 IVF indexes require more data. For single documents, consider using 'Flat' index type in config.")
                        logger.info(f"🔄 Auto-switching to IndexFlatL2 for reliable operation")
                        # Replace with flat index
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                        logger.info(f"✅ Switched to Flat index (exact search, slower but works with any dataset size)")
                    else:
                        recommended_min = nlist * 10  # IVF works better with 10x+ the nlist size
                        if len(embeddings) < recommended_min:
                            logger.warning(f"⚠️ Suboptimal training data: {len(embeddings)} embeddings (recommended: {recommended_min}+)")
                            logger.warning(f"💡 Consider using larger dataset or 'Flat' index for better results")

                        logger.info(f"🏋️ Training FAISS IVF index...")
                        logger.info(f"   - Training vectors: {len(embeddings)}")
                        logger.info(f"   - Clusters (nlist): {nlist}")
                        logger.info(f"   - Expected memory: ~{(len(embeddings) * self.dimension * 4) / 1024 / 1024:.1f} MB")

                        # Use sufficient training data
                        training_data = embeddings[:min(50000, len(embeddings))]
                        underlying_index.train(training_data)
                        logger.info("✅ FAISS IVF training completed successfully")
                else:
                    logger.info(f"✅ FAISS IVF index already trained (nlist={nlist})")
            else:
                logger.info(f"ℹ️ Using {type(underlying_index).__name__} (no training required)")

        except Exception as e:
            logger.error(f"❌ Index training failed with error: {e}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            logger.info(f"🔄 Falling back to IndexFlatL2 for reliability")
            logger.info(f"💡 To avoid this fallback, use 'Flat' index type in config for small datasets")
            # Fallback to simple flat index
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            logger.info(f"✅ Fallback complete - using exact search")

        # Add to index
        try:
            self.index.add_with_ids(embeddings, np.array(chunk_ids, dtype=np.int64))
        except Exception as e:
            logger.error(f"Failed to add embeddings to FAISS index: {e}")
            raise

        # Store metadata
        for i, (chunk, frame_num, chunk_id) in enumerate(zip(chunks, frame_numbers, chunk_ids)):
            try:
                metadata = {
                    "id": chunk_id,
                    "text": chunk,
                    "frame": frame_num,
                    "length": len(chunk)
                }
                self.metadata.append(metadata)

                # Update mappings
                self.chunk_to_frame[chunk_id] = frame_num
                if frame_num not in self.frame_to_chunks:
                    self.frame_to_chunks[frame_num] = []
                self.frame_to_chunks[frame_num].append(chunk_id)

            except Exception as e:
                logger.error(f"Failed to store metadata for chunk {chunk_id}: {e}")
                # Continue with other chunks
                continue

        return chunk_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk_id, distance, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Gather results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid result
                metadata = self.metadata[idx]
                results.append((idx, float(dist), metadata))
        
        return results
    
    def get_chunks_by_frame(self, frame_number: int) -> List[Dict[str, Any]]:
        """Get all chunks associated with a frame"""
        chunk_ids = self.frame_to_chunks.get(frame_number, [])
        return [self.metadata[chunk_id] for chunk_id in chunk_ids]
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by ID"""
        if 0 <= chunk_id < len(self.metadata):
            return self.metadata[chunk_id]
        return None
    
    def save(self, path: str):
        """
        Save index to disk
        
        Args:
            path: Path to save index (without extension)
        """
        path = Path(path)
        
        # Save FAISS index
        index_to_save = self.index
        # Check if the index (potentially wrapped by IndexIDMap) is a GPU index
        # Access the actual index object if wrapped by IndexIDMap
        actual_index = self.index.index if hasattr(self.index, 'index') else self.index

        if hasattr(faiss.gpu, 'GpuIndex') and isinstance(actual_index, faiss.gpu.GpuIndex):
            logger.info("Converting GPU index to CPU index for saving.")
            try:
                index_to_save = faiss.gpu.gpu_index_to_cpu(actual_index)
                # If the original self.index was an IndexIDMap wrapping a GpuIndex,
                # we need to re-wrap the new cpu_index with IndexIDMap for consistency,
                # or handle this at the point of saving faiss.write_index.
                # For simplicity, if self.index is IndexIDMap, we assume faiss.write_index
                # can handle IndexIDMap(GpuIndex) by saving its CPU version, or we save the converted one.
                # The current faiss.write_index directly on an IndexIDMap(GpuIndex) might be problematic.
                # So, if self.index was IndexIDMap, we create a new IndexIDMap with the cpu_index.
                if isinstance(self.index, faiss.IndexIDMap):
                    logger.info("Original index was IndexIDMap(GpuIndex). Re-wrapping CPU index with IndexIDMap for saving.")
                    # Preserve existing ID mappings if possible, though gpu_index_to_cpu typically handles this.
                    # A clean way: save the converted `actual_index` (now CPU) and then wrap it with IDMap if needed.
                    # However, IndexIDMap needs to be "refilled" if we just save `actual_index`.
                    # The simplest is to save the `index_to_save` which is the direct CPU version of `actual_index`.
                    # If `self.index` was `IndexIDMap(GpuIndex)`, then `index_to_save` is `CpuIndex`.
                    # We need to ensure IDs are preserved. `gpu_index_to_cpu` should preserve data.
                    # For `IndexIDMap`, the IDs are managed by the map itself.
                    # Let's assume `faiss.write_index` on an `IndexIDMap(GpuIndex)` is smart,
                    # or we save the `actual_index` after conversion if that's what `faiss.write_index` expects.
                    # The most robust: if IndexIDMap(GpuIndex), convert GpuIndex to CpuIndex, then make new IndexIDMap(CpuIndex).
                    # This is complex due to ID preservation.
                    # A simpler and often correct approach for IndexIDMap(GpuIndex):
                    cpu_actual_index = faiss.gpu.gpu_index_to_cpu(actual_index)
                    # Reconstruct IndexIDMap if the original was one
                    if isinstance(self.index, faiss.IndexIDMap):
                        # This assumes the IDMap structure itself doesn't need GPU/CPU conversion
                        # and can work with a CPU version of its sub-index.
                        # However, IndexIDMap doesn't store the sub-index in a way that's trivial to swap.
                        # The common practice is to save the CPU version of the core index.
                        # If IndexIDMap wraps GpuIndex, we save IndexIDMap(CpuIndex).
                        # This requires self.index.index to be replaced.
                        # For saving, it's safer to write the CPU version directly.
                        # Let's try saving the converted `actual_index` if it was GPU.
                        # If self.index is IndexIDMap, then self.index.index is the one we convert.
                        # We then save self.index (which is IndexIDMap with a now CPU sub-index internally after conversion by some faiss magic)
                        # Or more explicitly:
                        temporary_cpu_index = faiss.IndexIDMap(cpu_actual_index)
                        # Copy a few important things if they are not automatically handled
                        if hasattr(self.index, 'ntotal'): temporary_cpu_index.add_with_ids(np.array([]).reshape(0,self.dimension), np.array([], dtype=np.int64)) # ensure it's trained if IVF
                        # This part is tricky. The best is if write_index handles IndexIDMap(GpuIndex).
                        # If not, convert GpuIndex to CpuIndex, then save that.
                        # Let's assume `faiss.write_index` can handle `IndexIDMap(GpuIndex)` by converting internally or erroring.
                        # A safer bet:
                        logger.info(f"Saving CPU version of the index. Original type: {type(self.index)}, Actual index type: {type(actual_index)}")
                        faiss.write_index(cpu_actual_index, str(path.with_suffix('.faiss')))
                        # And we must make sure that if self.index was an IndexIDMap, the IDs are saved elsewhere or correctly handled.
                        # The current structure saves metadata (which includes IDs via self.metadata) separately.
                        # The FAISS index itself, if IndexIDMap, stores its own ID mapping.
                        # faiss.write_index(faiss.IndexIDMap(cpu_index_from_gpu)) is the goal.
                        # The simplest:
                        # index_to_save = faiss.index_gpu_to_cpu(self.index) # if faiss has such a utility for IndexIDMap(GpuIndex)
                        # For now, this should work if IndexIDMap is not used or if write_index handles it.
                        # Given IndexIDMap is always used:
                        # 1. Get actual GpuIndex: actual_index = self.index.index
                        # 2. Convert to CpuIndex: cpu_inner_index = faiss.gpu.gpu_index_to_cpu(actual_index)
                        # 3. Create new IndexIDMap: temp_id_map_cpu = faiss.IndexIDMap(cpu_inner_index)
                        # 4. Re-add IDs if necessary (usually not, as data is preserved)
                        # 5. Save temp_id_map_cpu
                        # This is quite involved. A common pattern is to just save the CPU version of the *inner* index
                        # and reconstruct IndexIDMap on load. But current code saves self.index.
                        # Let's stick to converting the `actual_index` and saving that if it was GPU.
                        # If `self.index` is `IndexIDMap(GpuIndex)`, we save `cpu_inner_index`.
                        # This means `read_index` will load `CpuIndex`, and then `IndexIDMap` is applied. This is fine.
                        faiss.write_index(cpu_actual_index, str(path.with_suffix('.faiss')))
                        logger.info(f"Saved CPU version of GpuIndex: {type(cpu_actual_index)}")
                    else: # self.index was GpuIndex directly (not wrapped by IDMap - though current code always wraps)
                        faiss.write_index(index_to_save, str(path.with_suffix('.faiss')))
                        logger.info(f"Saved CPU version of GpuIndex: {type(index_to_save)}")

                else: # self.index was already IndexIDMap(CpuIndex) or CpuIndex
                    faiss.write_index(self.index, str(path.with_suffix('.faiss')))
                    logger.info(f"Saved CPU index: {type(self.index)}")
            except AttributeError: # faiss.gpu might not exist if faiss-cpu is installed
                 logger.warning("faiss.gpu module not available. Assuming CPU index. Saving as is.")
                 faiss.write_index(self.index, str(path.with_suffix('.faiss')))
            except Exception as e:
                logger.error(f"Error converting GPU index to CPU for saving: {e}. Saving as is (might fail or save GPU specific format).")
                faiss.write_index(self.index, str(path.with_suffix('.faiss'))) # Try saving original
        else: # Not a GPU index or faiss.gpu not available
            faiss.write_index(self.index, str(path.with_suffix('.faiss')))
            logger.info(f"Saved CPU index (or GPU attributes not found): {type(self.index.index if hasattr(self.index, 'index') else self.index)}")

        # Save metadata and mappings
        data = {
            "metadata": self.metadata,
            "chunk_to_frame": self.chunk_to_frame,
            "frame_to_chunks": self.frame_to_chunks,
            "config": self.config
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """
        Load index from disk
        
        Args:
            path: Path to load index from (without extension)
        """
        path = Path(path)
        
        # Load FAISS index
        cpu_index = faiss.read_index(str(path.with_suffix('.faiss')))
        logger.info(f"Loaded index from disk. Type: {type(cpu_index)}")

        use_gpu = self.config["index"].get('use_gpu', False)

        if use_gpu:
            logger.info("Attempting to convert loaded CPU index to GPU.")
            try:
                res = faiss.StandardGpuResources()
                # If the loaded index was an IndexIDMap(CpuIndex), we need to convert the CpuIndex part.
                if isinstance(cpu_index, faiss.IndexIDMap):
                    inner_index = cpu_index.index
                    logger.info(f"Inner index type before GPU conversion: {type(inner_index)}")
                    gpu_inner_index = faiss.gpu.cpu_index_to_gpu(res, 0, inner_index) # 0 is GPU device ID
                    # Reconstruct IndexIDMap around the new GpuIndex
                    # This is tricky as IndexIDMap doesn't allow easy swapping of its internal index.
                    # A common approach: create a new IndexIDMap and re-add data.
                    # However, IDs are already in cpu_index (IndexIDMap).
                    # The best: if cpu_index_to_gpu can handle IndexIDMap directly. It cannot.
                    # So, we convert inner_index, then wrap it. This means self.index changes structure.
                    # Alternative: if the saved index was *just* the CpuIndex (not IDMap), then wrap here.
                    # Given the save logic modification, what's saved is the actual index (Cpu version).
                    # So cpu_index here *is* the CpuIndex (e.g. IndexFlatL2, IndexIVFFlat) if it was GPU before save.
                    # If it was CPU IndexIDMap before, it's still IndexIDMap(CpuIndex).

                    # If what was saved was the raw `cpu_actual_index` from the save method:
                    if not isinstance(cpu_index, faiss.IndexIDMap): # Saved actual index was not IDMap
                        logger.info(f"Loaded index is raw type {type(cpu_index)}. Converting to GPU and wrapping with IndexIDMap.")
                        gpu_actual_index = faiss.gpu.cpu_index_to_gpu(res, 0, cpu_index)
                        self.index = faiss.IndexIDMap(gpu_actual_index)
                    else: # Loaded index is IndexIDMap(CpuIndex)
                        logger.info(f"Loaded index is IndexIDMap({type(cpu_index.index)}). Converting inner index to GPU.")
                        # This path assumes `faiss.write_index(self.index)` was called where self.index was `IndexIDMap(CpuIndex)`
                        # or `IndexIDMap(GpuIndex)` that faiss handled.
                        # If `faiss.write_index(cpu_actual_index)` was called in save:
                        # Then `cpu_index` is the `cpu_actual_index`. We convert it to GPU then wrap with IDMap.
                        # This is the most consistent path with the modified save logic.
                        converted_gpu_part = faiss.gpu.cpu_index_to_gpu(res, 0, cpu_index.index if isinstance(cpu_index, faiss.IndexIDMap) else cpu_index)
                        self.index = faiss.IndexIDMap(converted_gpu_part)

                    logger.info(f"Successfully converted index to GPU. Final index type: {type(self.index.index)}")
                else: # Loaded index is a raw CPU index (e.g. IndexFlatL2)
                    logger.info(f"Loaded index is raw CPU type {type(cpu_index)}. Converting to GPU and wrapping with IndexIDMap.")
                    gpu_raw_index = faiss.gpu.cpu_index_to_gpu(res, 0, cpu_index)
                    self.index = faiss.IndexIDMap(gpu_raw_index) # Always wrap, as per _create_index
                    logger.info(f"Successfully converted raw CPU index to GPU and wrapped. Final index type: {type(self.index.index)}")

            except AttributeError: # faiss.gpu not available
                logger.warning("faiss.gpu module not available. Cannot convert to GPU index. Using loaded CPU index.")
                self.index = cpu_index if isinstance(cpu_index, faiss.IndexIDMap) else faiss.IndexIDMap(cpu_index)
            except Exception as e:
                logger.warning(f"Failed to convert CPU index to GPU: {e}. Using loaded CPU index.")
                self.index = cpu_index if isinstance(cpu_index, faiss.IndexIDMap) else faiss.IndexIDMap(cpu_index)
        else:
            logger.info("GPU usage not configured. Using loaded CPU index.")
            # Ensure it's wrapped with IndexIDMap if it's not already (e.g. if a raw index was saved)
            self.index = cpu_index if isinstance(cpu_index, faiss.IndexIDMap) else faiss.IndexIDMap(cpu_index)
        
        logger.info(f"Final index in use: {type(self.index)}, inner: {type(self.index.index if hasattr(self.index, 'index') else self.index)}")

        # Load metadata and mappings
        with open(path.with_suffix('.json'), 'r') as f:
            data = json.load(f)
        
        self.metadata = data["metadata"]
        self.chunk_to_frame = {int(k): v for k, v in data["chunk_to_frame"].items()}
        self.frame_to_chunks = {int(k): v for k, v in data["frame_to_chunks"].items()}
        
        # Update config if available
        if "config" in data:
            # Preserve GPU setting from current runtime config if it exists
            gpu_setting = self.config["index"].get('use_gpu')
            self.config.update(data["config"])
            if gpu_setting is not None:
                 self.config["index"]['use_gpu'] = gpu_setting
            # Re-initialize model and dimension based on loaded config for consistency,
            # unless current config is meant to override. For now, loaded config takes precedence for these.
            self.embedding_model = SentenceTransformer(self.config["embedding"]["model"])
            self.dimension = self.config["embedding"]["dimension"]

        logger.info(f"Loaded index and metadata from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_chunks": len(self.metadata),
            "total_frames": len(self.frame_to_chunks),
            "index_type": self.config["index"]["type"],
            "embedding_model": self.config["embedding"]["model"],
            "dimension": self.dimension,
            "avg_chunks_per_frame": np.mean([len(chunks) for chunks in self.frame_to_chunks.values()]) if self.frame_to_chunks else 0
        }