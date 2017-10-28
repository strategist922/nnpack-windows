from __future__ import absolute_import
from __future__ import division

def fp16_alt_xmm_to_fp32_xmm(xmm_half):
	xmm_zero = XMMRegister()
	VPXOR(xmm_zero, xmm_zero, xmm_zero)

	xmm_word = XMMRegister()
	VPUNPCKLWD(xmm_word, xmm_zero, xmm_half)

	xmm_shl1_half = XMMRegister()
	VPADDW(xmm_shl1_half, xmm_half, xmm_half)

	xmm_shl1_nonsign = XMMRegister()
	VPADDD(xmm_shl1_nonsign, xmm_word, xmm_word)

	sign_mask = Constant.float32x4(-0.0)

	xmm_sign = XMMRegister()
	VANDPS(xmm_sign, xmm_word, sign_mask)

	xmm_shr3_nonsign = XMMRegister()
	VPSRLD(xmm_shr3_nonsign, xmm_shl1_nonsign, 4)

	exp_offset = Constant.uint32x4(0x38000000)

	xmm_norm_nonsign = XMMRegister()
	VPADDD(xmm_norm_nonsign, xmm_shr3_nonsign, exp_offset)

	magic_mask = Constant.uint16x8(0x3E80)
	xmm_denorm_nonsign = XMMRegister()
	VPUNPCKLWD(xmm_denorm_nonsign, xmm_shl1_half, magic_mask)

	magic_bias = Constant.float32x4(0.25)
	VSUBPS(xmm_denorm_nonsign, xmm_denorm_nonsign, magic_bias)

	xmm_denorm_cutoff = XMMRegister()
	VMOVDQA(xmm_denorm_cutoff, Constant.uint32x4(0x00800000))
	
	xmm_denorm_mask = XMMRegister()
	VPCMPGTD(xmm_denorm_mask, xmm_denorm_cutoff, xmm_shr3_nonsign)

	xmm_nonsign = XMMRegister()
	VBLENDVPS(xmm_nonsign, xmm_norm_nonsign, xmm_denorm_nonsign, xmm_denorm_mask)

	xmm_float = XMMRegister()
	VORPS(xmm_float, xmm_nonsign, xmm_sign)

	return xmm_float


def fp16_alt_xmm_to_fp32_ymm(xmm_half):
	ymm_half = YMMRegister()
	VPERMQ(ymm_half, xmm_half.as_ymm, 0b01010000)

	ymm_zero = YMMRegister()
	VPXOR(ymm_zero.as_xmm, ymm_zero.as_xmm, ymm_zero.as_xmm)

	ymm_word = YMMRegister()
	VPUNPCKLWD(ymm_word, ymm_zero, ymm_half)

	ymm_shl1_half = YMMRegister()
	VPADDW(ymm_shl1_half, ymm_half, ymm_half)

	ymm_shl1_nonsign = YMMRegister()
	VPADDD(ymm_shl1_nonsign, ymm_word, ymm_word)

	sign_mask = Constant.float32x8(-0.0)

	ymm_sign = YMMRegister()
	VANDPS(ymm_sign, ymm_word, sign_mask)

	ymm_shr3_nonsign = YMMRegister()
	VPSRLD(ymm_shr3_nonsign, ymm_shl1_nonsign, 4)

	exp_offset = Constant.uint32x8(0x38000000)

	ymm_norm_nonsign = YMMRegister()
	VPADDD(ymm_norm_nonsign, ymm_shr3_nonsign, exp_offset)

	magic_mask = Constant.uint16x16(0x3E80)
	ymm_denorm_nonsign = YMMRegister()
	VPUNPCKLWD(ymm_denorm_nonsign, ymm_shl1_half, magic_mask)

	magic_bias = Constant.float32x8(0.25)
	VSUBPS(ymm_denorm_nonsign, ymm_denorm_nonsign, magic_bias)

	ymm_denorm_cutoff = YMMRegister()
	VMOVDQA(ymm_denorm_cutoff, Constant.uint32x8(0x00800000))
	
	ymm_denorm_mask = YMMRegister()
	VPCMPGTD(ymm_denorm_mask, ymm_denorm_cutoff, ymm_shr3_nonsign)

	ymm_nonsign = YMMRegister()
	VBLENDVPS(ymm_nonsign, ymm_norm_nonsign, ymm_denorm_nonsign, ymm_denorm_mask)

	ymm_float = YMMRegister()
	VORPS(ymm_float, ymm_nonsign, ymm_sign)

	return ymm_float


simd_width = YMMRegister.size // float_.size

for fusion_factor in range(1, 8 + 1):
	arg_x = Argument(ptr(const_float_), "x")
	arg_y = Argument(ptr(const_float_), "y")
	arg_stride_y = Argument(size_t, "stride_y")
	arg_sum = Argument(ptr(float_), "sum")
	arg_n = Argument(size_t, "n")
	with Function("nnp_shdotxf{fusion_factor}__avx2".format(fusion_factor=fusion_factor),
		(arg_x, arg_y, arg_stride_y, arg_sum, arg_n),
		target=uarch.default + isa.fma3 + isa.avx2):

		reg_x = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_x, arg_x)

		reg_ys = [GeneralPurposeRegister64() for m in range(fusion_factor)]
		LOAD.ARGUMENT(reg_ys[0], arg_y)

		reg_stride_y = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_stride_y, arg_stride_y)
		ADD(reg_stride_y, reg_stride_y)

		reg_sum = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_sum, arg_sum)

		reg_n = GeneralPurposeRegister64()
		LOAD.ARGUMENT(reg_n, arg_n)

		ymm_accs = [YMMRegister() for m in range(fusion_factor)]
		VZEROALL()

		for m in range(1, fusion_factor):
			LEA(reg_ys[m], [reg_ys[m - 1] + reg_stride_y * 1])

		main_loop = Loop()
		edge_loop = Loop()

		SUB(reg_n, XMMRegister.size // uint16_t.size)
		JB(main_loop.end)

		with main_loop:
			ymm_x = YMMRegister()
			VMOVUPS(ymm_x, [reg_x])
			ADD(reg_x, YMMRegister.size)

			for reg_y, ymm_acc in zip(reg_ys, ymm_accs):
				xmm_half = XMMRegister()
				VMOVUPS(xmm_half, [reg_y])
				ADD(reg_y, XMMRegister.size)

				ymm_y = fp16_alt_xmm_to_fp32_ymm(xmm_half)
				VFMADD231PS(ymm_acc, ymm_x, ymm_y)

			SUB(reg_n, YMMRegister.size // float_.size)
			JAE(main_loop.begin)

		ADD(reg_n, XMMRegister.size // uint16_t.size)
		JE(edge_loop.end)

		with edge_loop:
			xmm_x = XMMRegister()
			VMOVSS(xmm_x, [reg_x])
			ADD(reg_x, YMMRegister.size)

			for reg_y, ymm_acc in zip(reg_ys, ymm_accs):
				reg_half = GeneralPurposeRegister32()
				MOVZX(reg_half, word[reg_y])

				xmm_half = XMMRegister()
				VMOVD(xmm_half, reg_half)
				ADD(reg_y, uint16_t.size)

				ymm_y = fp16_alt_xmm_to_fp32_ymm(xmm_half)
				VFMADD231PS(ymm_acc, xmm_x.as_ymm, ymm_y)

			SUB(reg_n, 1)
			JAE(edge_loop.begin)

		# Reduce the SIMD registers into a single elements
		xmm_tmp = XMMRegister()
		for i, ymm_acc in enumerate(ymm_accs):
			VEXTRACTF128(xmm_tmp, ymm_acc, 1)
			VADDPS(ymm_acc.as_xmm, ymm_acc.as_xmm, xmm_tmp)
			VHADDPS(ymm_acc, ymm_acc, ymm_acc)
			VHADDPS(ymm_acc, ymm_acc, ymm_acc)
			VMOVSS([reg_sum + i * float_.size], ymm_acc.as_xmm)

		RETURN()

