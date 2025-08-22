/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/collective_builder_decl.hpp"
#include "cute/arch/mma_sm80.hpp"


namespace cutlass::gemm::collective {

template <
  class ElementA,
  class GmemLayoutPairA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutPairB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm89,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutPairA,
    AlignmentA,
    ElementB,
    GmemLayoutPairB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    Shape<_1,_1,_1>,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelMultistageFP8BlockScaledAccum>>
> {
  static constexpr int PipelineStages = 4;
  static constexpr int WARP_M = 2;
  static constexpr int WARP_N = 2;
  static constexpr int NUM_WARPS = WARP_M * WARP_N;
  static constexpr int NUM_THREADS = NUM_WARPS * 32;

  using GmemLayoutATag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairA{}))>;
  using GmemLayoutSFATag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairA{}))>;
  using GmemLayoutBTag   = cute::remove_cvref_t<decltype(get<0>(GmemLayoutPairB{}))>;
  using GmemLayoutSFBTag = cute::remove_cvref_t<decltype(get<1>(GmemLayoutPairB{}))>;
  static_assert(cute::depth(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 2 and 
                cute::depth(cute::remove_pointer_t<GmemLayoutSFBTag>{}) == 2, 
      "Expect SFA and SFB layout to be depth of two with shape ((SFVecMN, restMN),(SFVecK, restK), L)");
  static_assert(size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{}) == 
                size<1,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{}), 
      "SFA and SFB must have equivalent SF vector sizes along K");

  static constexpr auto ScaleGranularityM = size<0,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});
  static constexpr auto ScaleGranularityN = size<0,0>(cute::remove_pointer_t<GmemLayoutSFBTag>{});
  static constexpr auto ScaleGranularityK = size<1,0>(cute::remove_pointer_t<GmemLayoutSFATag>{});

  static_assert(is_static<TileShape_MNK>::value);
  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();
  static_assert(IsFP8Input, "Warp Specialized gemm with FP8 BlockScaled Accumulator is only compatible with FP8 Blocked Scaled version right now.");
  
  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;
  using ElementBlockScale = ElementAccumulator;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  using TiledMma = decltype(cute::make_tiled_mma(cute::SM89_16x8x32_F32E4M3E4M3F32_TN{}, Layout<Shape<Int<WARP_M>, Int<WARP_N>, _1>>{}));

  
  using AlignmentTypeA = cute::uint_byte_t<static_cast<int>(sizeof(ElementA)) * AlignmentA>;
  using GmemCopyAtomA = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeA>, ElementA>;
  using GmemTiledCopyA = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomA, NUM_THREADS, AlignmentA, TagToStrideA_t<GmemLayoutATag>,
      decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using AlignmentTypeB = cute::uint_byte_t<static_cast<int>(sizeof(ElementB)) * AlignmentB>;
  using GmemCopyAtomB = cute::Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<AlignmentTypeB>, ElementB>;
  using GmemTiledCopyB = decltype(detail::make_simt_gmem_tiled_copy<
      GmemCopyAtomB, NUM_THREADS, AlignmentB, TagToStrideB_t<GmemLayoutBTag>,
      decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());


  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementAMma>;
  using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementBMma>;

  static constexpr int ScaleMsPerTile = size<0>(TileShape_MNK{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape_MNK{}) / ScaleGranularityN;
  static_assert((size<0>(TileShape_MNK{}) % ScaleGranularityM) == 0, "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert((size<1>(TileShape_MNK{}) % ScaleGranularityN) == 0, "FP8 scaling granularity must evenly divide tile shape along N.");


  using DispatchPolicy = MainloopSm89CpAsyncBlockScalingFP8<
      PipelineStages>;

  using CollectiveOp = CollectiveMma<
    DispatchPolicy,
    TileShape_MNK,
    ElementA,
    cute::tuple<TagToStrideA_t<GmemLayoutATag>, TagToStrideA_t<GmemLayoutSFATag>>,
    ElementB,
    cute::tuple<TagToStrideB_t<GmemLayoutBTag>, TagToStrideB_t<GmemLayoutSFBTag>>,
    TiledMma,
    GmemTiledCopyA,
    SmemLayoutAtomA,
    SmemCopyAtomA,
    cute::identity,
    GmemTiledCopyB,
    SmemLayoutAtomB,
    SmemCopyAtomB,
    cute::identity
  >;
};

} // namespace cutlass::gemm::collective
