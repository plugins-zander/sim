var fp16, bf16, fp32, fp64, fp128;

/* Arbitrary precision integer arithmetic. Little endian order (first byte is LSB byte). */;

function BigNum(bytes, neg) {
	this.bytes = bytes || [];
	this.neg = neg ? true : false;
	return this._trunc();
}

BigNum.from_int32 = function (int32) {
	var res = BigNum.from_uint32(Math.abs(int32));
	res.neg = int32 < 0;
	return res._trunc();
}

BigNum.from_uint16 = function (uint32) {
	var res = new BigNum();
	res.set_byte(0, (uint32 >>> 0) & 0xff);
	res.set_byte(1, (uint32 >>> 8) & 0xff);
	return res._trunc();
}

BigNum.from_uint32 = function (uint32) {
	var res = new BigNum();
	res.set_byte(0, (uint32 >>> 0) & 0xff);
	res.set_byte(1, (uint32 >>> 8) & 0xff);
	res.set_byte(2, (uint32 >>> 16) & 0xff);
	res.set_byte(3, (uint32 >>> 24) & 0xff);
	return res._trunc();
}

BigNum.from_uint64 = function (uint32h, uint32l) {
	var res = new BigNum();
	res.set_byte(0, (uint32l >>> 0) & 0xff);
	res.set_byte(1, (uint32l >>> 8) & 0xff);
	res.set_byte(2, (uint32l >>> 16) & 0xff);
	res.set_byte(3, (uint32l >>> 24) & 0xff);
	res.set_byte(4, (uint32h >>> 0) & 0xff);
	res.set_byte(5, (uint32h >>> 8) & 0xff);
	res.set_byte(6, (uint32h >>> 16) & 0xff);
	res.set_byte(7, (uint32h >>> 24) & 0xff);
	return res._trunc();
}

BigNum.zero = function () { return new BigNum([0]); }
BigNum.one = function () { return new BigNum([1]); }

BigNum.from_string = function (str, base) {

	if (!str || str.length === 0)
		return undefined;

	var negative = false;

	if (str.charAt(0) === "-") {
		negative = true;
		str = str.substring(1);
	} else if (str.charAt(0) === "+") {
		str = str.substring(1);
	}

	var CHUNK_LENGTH = 8;

	var scale_factor;
	if (base === undefined || base == 10) {
		if (!/^\d+$/.test(str))
			return undefined;

		/* 10^8 */
		scale_factor = BigNum.from_uint32(100000000);
		base = 10;
	} else if (base === 16) {
		str = str.toLowerCase();
		if (str.length > 2 && str.charAt(0) === "0" && str.charAt(1) === "x")
			str = str.substring(2);

		if (!/^[0-9a-f]+$/.test(str))
			return undefined;

		/* Each hex digit is 4 bits, so 8 digits in chunks is 32 bits. */
		scale_factor = BigNum.one().shl(32);
		base = 16;
	}

	var res = BigNum.zero();

	/* Split string from the tail into chunks that can be easily parsed with native
		functions then just keep adding the chunk value multiplied by it's scale */

	var chunks = [];
	for (var i = str.length; i > 0; i -= CHUNK_LENGTH)
		chunks.push(str.substring(Math.max(0, i - CHUNK_LENGTH), i))

	var scale = BigNum.from_uint32(1);
	for (var i = 0; i < chunks.length; i++) {
		var num = BigNum.from_uint32(parseInt(chunks[i], base));
		res = res.add(num.mul(scale));
		scale = scale.mul(scale_factor);
	}

	res.neg = negative
	return res._trunc();
}

BigNum.prototype.toString = function (base) {

	if (this.bytes.length == 0)
		return "0";

	var components = [];

	if (base == 2) {
		for (var i = this.bit_length() - 1; i >= 0; i--) {
			components.push(this.bit(i));
		}
	} else if (base === undefined || base == 10) {
		var res = [0];

		var length = this.bit_length();

		for (var i = 0; i < length; i++) {

			if (this.bit(i) != 0) {

				var exp = BigNum._pow2_digits_base_big10(i);
				var carry = 0;

				for (var j = 0; j < res.length; j++) {
					var val = res[j] + exp[j] + carry;
					carry = Math.floor(val / BigNum._big10);
					res[j] = (val - carry * BigNum._big10);
				}

				for (var j = res.length; j < exp.length; j++) {
					var val = exp[j] + carry;
					carry = Math.floor(val / BigNum._big10);
					res.push(val - carry * BigNum._big10);
				}

				if (carry)
					res.push(carry);
			}
		}

		var digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

		var res2 = [];


		for (var i = 0; i < res.length; i++) {
			var chunk = res[i];
			var d = 0;
			while (chunk > 0) {
				var next = Math.floor(chunk / 10.0);
				res2.push(digits[chunk - next*10]);
				chunk = next;
				d++;
			}

			if (i != res.length - 1) {
				for (; d < BigNum._big10exp; d++)
					res2.push("0");
			}	
		}	
		components = res2.reverse();		
	} else if (base == 16) {
		var skipped_zeros = false;

		for (var i = this.bytes.length - 1; i >= 0; i--) {
			var byte = this.bytes[i];
			for (var j = 0; j < 2; j++) {
				var nibble = (byte & 0xf0) >>> 4;

				if (nibble != 0 || skipped_zeros) {
					skipped_zeros = true;
					components.push(nibble.toString(16));
				}
				byte <<= 4;
			}
		}
	}

	var sign_string = this.neg ? "-" : "";
	return sign_string + components.join("");
}

BigNum.prototype.copy = function () { return new BigNum(this.bytes.slice(), this.neg); }

BigNum.prototype.cmp = function (a) {
	if (this.neg !== a.neg)
		return a.neg ? +1 : -1;

	var gr = this.neg ? -1 : +1;
	var le = this.neg ? +1 : -1;
	if (this.bytes.length != a.bytes.length)
		return this.bytes.length > a.bytes.length ? gr : le;

	for (i = this.bytes.length - 1; i >= 0; i--) {
		if (this.bytes[i] != a.bytes[i])
			return this.bytes[i] > a.bytes[i] ? gr : le;
	}

	return 0;
}

BigNum.prototype.eq = function (a) {
	if (this.bytes.length != a.bytes.length)
		return false;

	for (i = 0; i < this.bytes.length; i++) {
		if (this.bytes[i] != a.bytes[i])
			return false;
	}

	return true;
}

BigNum.prototype.is_0 = function () {
	return this.bytes.length == 0;
}

BigNum.prototype.is_1 = function () {
	return this.bytes.length == 1 && this.bytes[0] == 1 && !this.neg;
}

BigNum.prototype.double_value = function () {
	var scale = 1.0;
	var val = 0.0;
	for (var i = 0; i < this.bytes.length; i++) {
		val += this.bytes[i] * scale;
		scale *= 256.0;
	}

	return this.neg ? -val : val;
}

BigNum.prototype.byte = function (i) {
	return i >= this.bytes.length ? 0 : this.bytes[i];
}


BigNum.prototype.set_byte = function (i, val) {
	while (this.bytes.length <= i) {
		this.bytes.push(0);
	}
	this.bytes[i] = val;

	return this._trunc();
}



BigNum.prototype.bit = function (i) {
	if (i >= this.bytes.length * 8)
		return 0;

	return ((this.bytes[i >> 3] >>> (i & 7))) & 1;
}


BigNum.prototype.set_bit = function (i, val) {
	while (this.bytes.length <= (i >>> 3)) {
		this.bytes.push(0);
	}

	val = val > 0 ? 1 : 0;
	this.bytes[i >> 3] &= ~(1 << (i & 7));
	this.bytes[i >> 3] |= val << (i & 7);

	return this._trunc();
}

BigNum.prototype.bit_length = function () {
	if (this.bytes.length == 0)
		return 0;

	var length = this.bytes.length * 8;
	var i;
	for (i = 7; i >= 0; i--) {
		if ((this.bytes[this.bytes.length - 1] & (1 << i)) != 0)
			break;
	}

	return length - 7 + i;
}

BigNum.prototype.pop_count = function () {
	if (this.bytes.length == 0)
		return 0;

	var count = 0;

		for (i = 0; i < this.bytes.length; i++) {
			var byte = this.bytes[i]
			count += (byte & (1<<0)) ? 1 : 0;
			count += (byte & (1<<1)) ? 1 : 0;
			count += (byte & (1<<2)) ? 1 : 0;
			count += (byte & (1<<3)) ? 1 : 0;
			count += (byte & (1<<4)) ? 1 : 0;
			count += (byte & (1<<5)) ? 1 : 0;
			count += (byte & (1<<6)) ? 1 : 0;
			count += (byte & (1<<7)) ? 1 : 0;
		}	

	return count;
}




BigNum.prototype.negated = function () {
	return new BigNum(this.bytes, !this.neg);
}

BigNum.prototype.abs = function () {
	return new BigNum(this.bytes, false);
}

BigNum.prototype.pow = function (a) {
	if (a === 0)
		return BigNum.one();

	var res = BigNum.one();

	if (typeof a === "number") {
		var val = this;
		while (a > 1) {
			if (a & 1) {
				res = res.mul(val);
				a--;
			}
			a >>>= 1;
			val = val.mul(val);
		}

		return res.mul(val)._trunc();
	}

	var base_x = this;

	if (a.bytes.length > 2)
		throw "Too big exponent!";

	if (a.bytes.length > 0)
		res = res.mul(base_x.pow(a.bytes[0]));
	if (a.bytes.length > 1)
		res = res.mul(base_x.pow(256).pow(a.bytes[1]));

	return res;
};


BigNum.prototype.mul = function (a) {

	if (typeof a === "number")
		a = BigNum.from_int32(a);

	if (this.is_1())
		return a.copy();

	if (a.is_1())
		return this.copy();

	if (this.is_0() || a.is_0())
		return BigNum.zero();

	var i, j;
	var res = new BigNum();

	for (i = 0; i < this.bytes.length; i++) {
		for (j = 0; j < a.bytes.length; j++) {
			var val = this.byte(i) * a.byte(j);

			/* Add in place, much faster than just using BigNum.add */
			for (var k = i + j; val !== 0; k++) {
				val += res.byte(k);
				res.set_byte(k, val & 0xff);
				val >>>= 8;
			}
		}
	}

	res.neg = a.neg !== this.neg;

	return res._trunc();
};

BigNum.prototype.add = function (a) {

	if (typeof a === "number")
		a = BigNum.from_int32(a);

	if (this.is_0())
		return a.copy();

	if (a.is_0())
		return this.copy();

	if (a.neg !== this.neg)
		return a.neg ? this.sub(a.negated()) : a.sub(this.negated());

	var val = 0;
	var res = new BigNum();
	var i = 0;
	for (i = 0; i < Math.max(this.bytes.length, a.bytes.length); i++) {
		val = this.byte(i) + a.byte(i) + (val >>> 8);
		res.set_byte(i, val & 0xff);
	}

	if (val != 0)
		res.set_byte(i, (val >>> 8));

	res.neg = a.neg;

	return res._trunc();
};

BigNum.prototype.sub = function (a) {

	if (typeof a === "number")
		a = BigNum.from_int32(a);
	// + -
	if (!this.neg && a.neg)
		return this.add(a.negated());

	// - +
	if (this.neg && !a.neg)
		return this.add(a.negated());

	var cmp = this.abs().cmp(a.abs());
	// + +
	if (!this.neg && !a.neg) {
		if (cmp == -1) {
			return a.sub(this).negated();
		}
	}

	if (this.neg && a.neg) {
		if (cmp == -1) {
			return a.sub(this);
		}
	}

	if (a.bytes.length == this.bytes.length) {
		if (a.bytes.length == 0)
			return this.copy();
		else if (a.bytes[a.bytes.length - 1] > this.bytes[this.bytes.length - 1])
			throw "Subtracting larger number!";
	}

	var carry = 0;
	var res = new BigNum();

	for (var i = 0; i < this.bytes.length; i++) {
		val = this.byte(i) - a.byte(i) - carry;
		carry = 0;
		if (val < 0) {
			val += 0x100;
			carry = 1;
		}
		res.set_byte(i, val);
	}

	return res._trunc();
};

BigNum.prototype.or = function (a) {

	if (typeof a === "number")
		a = BigNum.from_int32(a);

	if (this.neg || a.neg)
		throw "OR-ing negative numbers";

	var res = new BigNum();
	for (var i = 0; i < Math.max(this.bytes.length, a.bytes.length); i++)
		res.set_byte(i, this.byte(i) | a.byte(i));

	return res._trunc();
};

BigNum.prototype.and = function (a) {
	if (typeof a === "number")
		a = BigNum.from_int32(a);

	if (this.neg || a.neg)
		throw "AND-ing negative numbers";

	var res = new BigNum();
	for (var i = 0; i < Math.max(this.bytes.length, a.bytes.length); i++)
		res.set_byte(i, this.byte(i) & a.byte(i));

	return res._trunc();
};

BigNum.prototype.not = function (min_length) {

	if (this.neg)
		throw "NOT-ing negative number";

	var res = new BigNum();
	for (var i = 0; i < this.bytes.length; i++)
		res.set_byte(i, ~this.byte(i));

	for (var i = this.bytes.length; i < min_length; i++)
		res.set_byte(i, 0xff);

	return res._trunc();
};


BigNum.prototype.shl = function (shift) {

	var res = new BigNum();

	if (shift == 0)
		return this.copy();

	if (shift < 0)
		return this.shr(-shift);

	var byte_shift = shift >>> 3;
	shift -= byte_shift * 8;

	if (byte_shift != 0) {
		for (var i = 0; i < this.bytes.length; i++)
			res.set_byte(i + byte_shift, this.byte(i));
	} else {
		res = this.copy();
	}

	if (shift != 0) {
		var length = res.bytes.length;

		for (var i = byte_shift; i < length; i++)
			res.set_byte(i, res.byte(i) << shift);

		for (var i = byte_shift; i < length; i++)
			res.set_byte(i + 1, res.byte(i + 1) | (res.byte(i) >>> 8));

		for (var i = byte_shift; i < res.bytes.length; i++)
			res.set_byte(i, res.byte(i) & 0xff);
	}

	res.sign = this.sign;

	return res._trunc();
};

BigNum.prototype.shr = function (shift) {

	if (shift == 0)
		return this.copy();

	if (shift < 0)
		return this.shl(-shift);

	var byte_shift = shift >>> 3;

	if (this.bit_length() <= shift)
		return BigNum.zero();

	var res = new BigNum();

	shift -= byte_shift * 8;

	for (var i = 0; i < this.bytes.length - byte_shift; i++)
		res.set_byte(i, this.byte(i + byte_shift));

	for (var i = 0; i < res.bytes.length; i++)
		res.set_byte(i, res.byte(i) << (8 - shift));

	for (var i = 0; i < res.bytes.length; i++)
		res.set_byte(i, (res.byte(i) >>> 8) | (res.byte(i + 1) & 0xff));

	for (var i = byte_shift; i < res.bytes.length; i++)
		res.set_byte(i, res.byte(i) & 0xff);

	res.sign = this.sign;

	return res._trunc();
};



BigNum.prototype.toBigNum10 = function () {
	var res = [0];

	var length = this.bit_length();

	for (var i = 0; i < length; i++) {

		if (this.bit(i) != 0) {

			var exp = BigNum._pow2_digits_base_big10(i);
			var carry = 0;

			for (var j = 0; j < res.length; j++) {
				var val = res[j] + exp[j] + carry;
				carry = Math.floor(val / BigNum._big10);
				res[j] = (val - carry * BigNum._big10);
			}

			for (var j = res.length; j < exp.length; j++) {
				var val = exp[j] + carry;
				carry = Math.floor(val / BigNum._big10);
				res.push(val - carry * BigNum._big10);
			}

			if (carry)
				res.push(carry);
		}
	}

	var digits = new Array(res.length * BigNum._big10exp);

	for (var i = 0; i < res.length; i++) {
		var val = res[i];

		for (var j = 0; j < BigNum._big10exp; j++) {
			var trunc = Math.floor(val / 10);
			digits[BigNum._big10exp * i + j] = val - 10.0 * trunc;
			val = trunc;
		}
	}

	return new BigNum10(digits);
}

BigNum.pow5 = function (a) {

	if (!BigNum._pow5)
		BigNum._pow5 = [BigNum.one()];

	while (BigNum._pow5.length <= a)
		BigNum._pow5.push(BigNum._pow5[BigNum._pow5.length - 1].mul(5));

	return BigNum._pow5[a];
}

BigNum.pow10 = function (a) {

	if (!BigNum._pow10)
		BigNum._pow10 = [BigNum.one()];

	while (BigNum._pow10.length <= a)
		BigNum._pow10.push(BigNum._pow10[BigNum._pow10.length - 1].mul(10));

	return BigNum._pow10[a];
}

BigNum._big10exp = 14
BigNum._big10 = 1e14;

BigNum._pow2_digits_base_big10 = function (a) {

	if (!BigNum._pow2)
		BigNum._pow2 = [[1]];

	if (BigNum._pow2.length <= a) {
		for (var i = BigNum._pow2.length - 1; i < a; i++) {
			var carry = 0;
			var base = BigNum._pow2[i];
			var next = [];
			for (var j = 0; j < base.length; j++) {
				var value = base[j] * 2 + carry;
				carry = Math.floor(value / BigNum._big10);
				next.push(value - carry * BigNum._big10);
			}

			if (carry)
				next.push(carry);

			BigNum._pow2.push(next);
		}
	}

	return BigNum._pow2[a];
}

BigNum.prototype._trunc = function () {
	while (this.bytes.length > 0 && this.bytes[this.bytes.length - 1] == 0)
		this.bytes.pop();

	if (this.bytes.length == 0)
		this.neg = false;

	return this;
}

/* Arbitrary precision integer arithmetic in base-10, mostly for
	decimal string conversions. */

function BigNum10(digits) {
	this.digits = digits || [];
	return this._trunc();
}

BigNum10.prototype._trunc = function () {
	while (this.digits.length > 0 && this.digits[this.digits.length - 1] == 0)
		this.digits.pop();

	return this;
}

BigNum10.prototype.mul_int = function (mul) {

	var res = new BigNum10();
	var carry = 0;

	for (var i = 0; i < this.digits.length; i++) {
		var value = this.digits[i] * mul + carry;
		carry = Math.floor(value / 10);
		res.digits.push(value - carry * 10);
	}

	if (carry)
		res.digits.push(carry);

	return res;
}
BigNum10.prototype.add = function add(a) {
	var res = new BigNum10();

	var arr0 = this.digits;
	var arr1 = a.digits;
	var l0 = arr0.length;
	var l1 = arr1.length;
	var l = Math.min(l0, l1);

	var carry = 0;

	for (var i = 0; i < l; i++) {
		var val = arr0[i] + arr1[i] + carry;
		carry = Math.floor(val / 10);
		res.digits.push(val - carry * 10);
	}

	var arr = l1 > l0 ? arr1 : arr0;

	for (var i = l; i < arr.length; i++) {
		var val = arr[i] + carry;
		carry = Math.floor(val / 10);
		res.digits.push(val - carry * 10);
	}

	if (carry)
		res.digits.push(carry);

	return res;
}

fp16 = new FloatConfig(16, 10);
bf16 = new FloatConfig(16, 7);
fp32 = new FloatConfig(32, 23);
fp64 = new FloatConfig(64, 52);
fp128 = new FloatConfig(128, 112);

var string_to_config = {
	fp16: fp16,
	bf16: bf16,
	fp32: fp32,
	fp64: fp64,
	fp128: fp128,
};


function FloatConfig(total_width, mant_width) {
	this.total_width = total_width;
	this.exp_width = total_width - 1 - mant_width;
	this.mant_width = mant_width;

	this.plus_inf_value = BigNum.one().shl(this.exp_width).sub(BigNum.one()).shl(mant_width);
	this.minus_inf_value = this.plus_inf_value.or(BigNum.one().shl(total_width - 1));
	this.nan_value = this.plus_inf_value.or(BigNum.one().shl(mant_width - 1));

	this.max_exp = BigNum.one().shl(this.exp_width).sub(BigNum.one());
	this.max_mant = BigNum.one().shl(this.mant_width).sub(BigNum.one());
	this.max_val = BigNum.one().shl(this.total_width).sub(BigNum.one());

	this.max_biased_exp = BigNum.one().shl(this.exp_width - 1).sub(BigNum.one())
	this.min_biased_exp = this.max_biased_exp.negated();

	this.mant_integer_scale = BigNum.pow5(mant_width);

	this.max_digits = 21;

	this.name = "fp" + total_width;
}

function Float(big, config) {

	this._big = big;
	this._config = config;

	this._updated = false;

	return this;
}

Float.prototype._update = function () {

	if (this._updated)
		return;

	this._updated = true;
	this._sign = this._big.shr(this._config.total_width - 1);
	this._exp = this._big.shr(this._config.mant_width).and(BigNum.one().shl(this._config.exp_width).sub(BigNum.one()));
	this._mant = this._big.and(BigNum.one().shl(this._config.mant_width).sub(BigNum.one()));

	this._bias = BigNum.one().shl(this._config.exp_width - 1).sub(BigNum.one()).sub(BigNum.from_uint32(this._exp.is_0() ? 1 : 0));
}

Float.prototype.bit = function (i) {
	return this._big.bit(i);
}

Float.prototype.set_bit = function (i, val) {
	this._big.set_bit(i, val);
	this._updated = false;
}

Float.prototype.value = function () {
	return this._big;
}

Float.prototype.set_value = function (v) {
	this._big = v;
	this._updated = false;
}

Float.prototype.sign = function () {
	this._update();

	return this._sign;
}

Float.prototype.set_sign = function (sign) {
	this._big = this._big.and(BigNum.one().shl(this._config.total_width - 1).sub(BigNum.one()));
	this._big = this._big.or(sign.shl(this._config.total_width - 1));

	this._updated = false;
}


Float.prototype.exp = function () {
	this._update();

	return this._exp;
}

Float.prototype.biased_exp = function () {
	this._update();

	return this._exp.sub(this._bias);
}

Float.prototype.bias = function () {
	this._update();

	return this._bias;
}

Float.prototype.set_exp = function (exp) {
	this._big = this._big.and(BigNum.one().shl(this._config.exp_width).sub(BigNum.one()).shl(this._config.mant_width).not(this._config.total_width >>> 3));
	this._big = this._big.or(exp.shl(this._config.mant_width));

	this._updated = false;
}

Float.prototype.mant = function () {
	this._update();

	return this._mant;
}

Float.prototype.set_mant = function (mant) {
	this._big = this._big.and(BigNum.one().shl(this._config.mant_width).sub(BigNum.one()).not(this._config.total_width >>> 3));
	this._big = this._big.or(mant);

	this._updated = false;
}

Float.prototype.implicit_bit = function () {
	this._update();

	return this._exp.is_0() ? 0 : 1;
}





Float.prototype.to_string = function () {
	this._update();

	if (this._exp.eq(this._config.max_exp)) {
		if (this._mant.is_0())
			return this._sign.is_0() ? "Infinity" : "-Infinity";
		return "NaN"
	}

	if (this._exp.is_0() && this._mant.is_0()) {
		return this._sign.is_0() ? "0.0" : "-0.0";
	}

	var big_exp = this._integer_decimal_exp();
	var big = big_exp[0];
	var decimal_exp = big_exp[1];

	var res = big.toBigNum10();

	if (res.digits.length > this._config.max_digits) {
		for (var i = 0; i < res.digits.length - this._config.max_digits - 1; i++)
			res.digits[i] = 0;

		if (res.digits[res.digits.length - this._config.max_digits - 1] >= 5) {
			var diff = res.digits.slice(0, res.digits.length - this._config.max_digits);
			diff[res.digits.length - this._config.max_digits - 1] = 10 - diff[res.digits.length - this._config.max_digits - 1];
			res = res.add(new BigNum10(diff));
		}

		res.digits[res.digits.length - this._config.max_digits - 1] = 0;
	}


	var str = res.digits.reverse().join("");

	for (var i = str.length; i > 0; i--) {
		if (str.charAt(i - 1) !== "0")
			break;

		decimal_exp++;
	}

	str = str.substring(0, i);

	var k = str.length;
	var n = decimal_exp + k;

	var sign_str = this._sign.is_1() ? "-" : "";

	n_zeroes = function (n) {
		return (new Array(n + 1)).join("0");
	};

	/* https://www.ecma-international.org/ecma-262/6.0/#sec-tostring */
	if (k <= n && n <= this._config.max_digits) {
		return sign_str + str + n_zeroes(n - k) + ".0";
	} else if (0 < n && n <= this._config.max_digits) {
		return sign_str + str.substring(0, n) + "." + str.substring(n);
	} else if (-6 < n && n <= 0) {
		return sign_str + "0." + n_zeroes(-n) + str;
	} else if (k == 1) {
		return sign_str + str + ".0e" + (n - 1 > 0 ? "+" : "-") + Math.abs(n - 1).toString();
	}

	return sign_str + str.charAt(0) + "." + str.substring(1) + "e" + (n - 1 > 0 ? "+" : "-") + Math.abs(n - 1).toString();
}


Float.from_components = function (sign, exp, mant, config) {
	if (typeof sign === "number")
		sign = BigNum.from_uint32(sign);

	if (typeof exp === "number")
		exp = BigNum.from_uint32(exp);

	if (typeof mant === "number")
		mant = BigNum.from_uint32(mant);

	return new Float(sign.shl(config.total_width - 1).or(exp.shl(config.mant_width).or(mant)), config);
}

Float.from_string = function (str, config) {
	str = str.trim();

	if (str.length === 0)
		return undefined;

	str = str.toLowerCase();

	if (str === "m_e") return Float.from_string("2.71828182845904523536028747135266250", config);
	else if (str === "m_log2e") return Float.from_string("1.44269504088896340735992468100189214", config);
	else if (str === "m_log10e") return Float.from_string("0.434294481903251827651128918916605082", config);
	else if (str === "m_ln2") return Float.from_string("0.693147180559945309417232121458176568", config);
	else if (str === "m_ln10") return Float.from_string("2.30258509299404568401799145468436421", config);
	else if (str === "m_pi") return Float.from_string("3.14159265358979323846264338327950288", config);
	else if (str === "m_pi_2") return Float.from_string("1.57079632679489661923132169163975144", config);
	else if (str === "m_pi_4") return Float.from_string("0.785398163397448309615660845819875721", config);
	else if (str === "m_1_pi") return Float.from_string("0.318309886183790671537767526745028724", config);
	else if (str === "m_2_pi") return Float.from_string("0.636619772367581343075535053490057448", config);
	else if (str === "m_2_sqrtpi") return Float.from_string("1.12837916709551257389615890312154517", config);
	else if (str === "m_sqrt2") return Float.from_string("1.41421356237309504880168872420969808", config);
	else if (str === "m_sqrt1_2") return Float.from_string("0.707106781186547524400844362104849039", config);

	if (str == "nan")
		return new Float(config.nan_value, config);

	if (str == "inf" || str == "+inf" || str == "infinity" || str == "+infinity")
		return new Float(config.plus_inf_value, config);

	if (str == "-inf" || str == "-infinity")
		return new Float(config.minus_inf_value, config);

	var parsed = parseFloat(str);
	if (parsed === undefined || isNaN(parsed))
		return undefined;


	var buffer = new ArrayBuffer(8);
	var uint8View = new Uint8Array(buffer);
	var float64View = new Float64Array(buffer);

	float64View[0] = parsed;

	var array = Array.prototype.slice.call(uint8View);
	var float64 = new Float(new BigNum(array), fp64);

	if (config === fp64)
		return float64;

	var float = new Float(BigNum.zero(), config);

	float.set_sign(float64.sign());

	var biased_exp = float64.biased_exp();
	if (biased_exp.cmp(config.max_biased_exp) > 0) {
		float.set_exp(config.max_exp);

		if (float64.exp().eq(fp64.max_exp) && !float64.mant().is_0())
			float = config.nan_value;
	} else {

		var shift = fp64.mant_width - config.mant_width;
		var big_mant = float64.mant();
		var mant = big_mant.shr(shift);
		var trimmed = big_mant.sub(mant.shl(shift));

		var cmp = trimmed.cmp(BigNum.one().shl(shift - 1));
		if (cmp > 0 || (cmp == 0 && mant.bit(0) == 1))
			mant = mant.add(1);

		/* We may have overflown the mantissa and denormalized it. Check if all the bits 
		in mantissa place are 0 and fix up the exponent. */
		if (mant.bit(config.mant_width) == 1) {
			biased_exp = biased_exp.add(1);
			mant = BigNum.zero();
		}

		var exp;

		if (biased_exp.cmp(config.max_biased_exp) > 0) {
			exp = config.max_exp;
		} else if (biased_exp.cmp(config.min_biased_exp) <= 0) {
			exp = BigNum.zero();

			shift = config.min_biased_exp.double_value() - biased_exp.double_value() + 1;

			mant = mant.or(BigNum.one().shl(config.mant_width));
			big_mant = mant;
			mant = mant.shr(shift);
			trimmed = big_mant.sub(mant.shl(shift));

			var cmp = trimmed.cmp(BigNum.one().shl(shift - 1));
			if (cmp > 0 || (cmp == 0 && mant.bit(0) == 1))
				mant = mant.add(1);
		} else {
			exp = biased_exp.add(config.max_biased_exp);
		}

		float.set_exp(exp);
		float.set_mant(mant);
	}

	return float;
}


Float.prototype.to_exact_string = function () {
	this._update();

	if (this._exp.eq(this._config.max_exp)) {
		if (this._mant.is_0())
			return this._sign.is_0() ? "&#x221e;" : "&minus;&#x221e;";
		return "Not a Number"
	}

	if (this._exp.is_0() && this._mant.is_0()) {
		return this._sign.is_0() ? "0.0" : "&minus;0.0";
	}

	var big_exp = this._integer_decimal_exp();

	var str = big_exp[0].toString();
	var decimal_exp = big_exp[1];

	for (var i = str.length; i > 0; i--) {
		if (str.charAt(i - 1) !== "0")
			break;
	}
	decimal_exp += str.length - i;
	str = str.substring(0, i);
	decimal_exp += str.length - 1;

	if (str.length == 1)
		str += ".0";
	else
		str = str.substring(0, 1) + "." + str.substring(1);

	str += "&#8203;&#215;10<sup>" + decimal_exp.toString() + "</sup>";

	var sign_str = this._sign.is_1() ? "-" : "";

	return sign_str + str;
}

Float.prototype._integer_decimal_exp = function () {
	this._update();

	var shifted_exp = this.biased_exp().double_value();

	var val = this._mant;
	if (!this._exp.is_0())
		val = val.add(BigNum.from_uint32(1).shl(this._config.mant_width));

	val = val.mul(this._config.mant_integer_scale);

	var decimal_exp = -this._config.mant_width;

	if (shifted_exp < 0) {
		/* 1/2^x = (1/2^x) * (5^x/5^x) = 5^x/10^x */

		decimal_exp += shifted_exp;
		shifted_exp = Math.abs(shifted_exp);
		val = val.mul(BigNum.pow5(shifted_exp));
	} else {
		val = val.shl(shifted_exp);
	}

	return [val, decimal_exp];
}


Float.prototype.prev_next_delta_equal = function () {
	return (!this._mant.is_0() && (!this._mant.eq(this._config.max_mant) || !this._exp.add(BigNum.one()).eq(this._config.max_exp)))
		|| this._exp.eq(this._config.max_exp)
		|| this._exp.is_0()
		|| (this._mant.is_0() && this._exp.is_1());
}


Float.prototype.next_delta_string = function () {
	if (this._sign.is_0() && this._exp.add(1).eq(this._config.max_exp) && this._mant.eq(this._config.max_mant))
		return "&#x221e;";

	if (this._exp.eq(this._config.max_exp))
		return "Indeterminate";

	var delta = new Float(BigNum.zero(), this._config);
	var final_exp = this._exp.double_value() - this._config.mant_width;

	if (final_exp < 0)
		delta.set_mant(BigNum.one().shl(this._config.mant_width + final_exp - (this._exp.is_0() ? 0 : 1)));
	else
		delta.set_exp(BigNum.from_uint32(final_exp));

	return delta.to_exact_string();
}

Float.prototype.prev_delta_string = function () {
	if (this._sign.is_1() && this._exp.add(1).eq(this._config.max_exp) && this._mant.eq(this._config.max_mant))
		return "&minus;&#x221e;";

	if (this._exp.eq(this._config.max_exp))
		return "Indeterminate";

	var delta = new Float(BigNum.zero(), this._config);
	var final_exp = this._exp.double_value() - this._config.mant_width;
	if (this._mant.is_0() && (!this._exp.is_0() || !this._exp.add(1).eq(this._config.max_exp)))
		final_exp--;

	if (final_exp < 0)
		delta.set_mant(BigNum.one().shl(this._config.mant_width + final_exp - (this._exp.is_0() ? 0 : 1)));
	else
		delta.set_exp(BigNum.from_uint32(final_exp));

	delta.set_sign(BigNum.one());
	return delta.to_exact_string();
}

Float.prototype.percent_a_string = function () {

	if (this._exp.eq(this._config.max_exp)) {
		if (this._mant.is_0())
			return this._sign.is_0() ? "inf" : "-inf";
		return "nan"
	}

	var ret = this._sign.is_1() ? "-" : "";

	if (this._exp.is_0()) {
		if (this._mant.is_0()) {
			ret += "0x0p+0";
		}
		else {
			var log = this._mant.bit_length();

			if (this._mant.pop_count() == 1) {
				ret += "0x1";
			}
			else {

				ret += "0x1.";

				var val = this._mant.and(BigNum.one().shl(log).not(log));

				var width = this._config.mant_width;
				var padded_width = 4 * Math.ceil(width / 4);

				var val = this._mant.shl(padded_width - width);


				for (var i = padded_width - 4; i >= 0; i -= 4) {
					var byte = val.shr(i).double_value();
					ret += byte.toString(16);

					val = val.and(BigNum.from_int32(15).shl(i).not(padded_width));

					if (val.is_0())
						break;
				}

			}

			var exp = "p" + (log - this._bias.double_value() - this._config.mant_width - 1).toString(10);

			ret += exp;

		}
	}
	else {
		var exp = this.biased_exp().double_value();

		if (this._mant.is_0()) {
			ret += "0x1p" + (exp >= 0 ? "+" : "") + exp.toString(10);
		}
		else {
			var width = this._config.mant_width;
			var padded_width = 4 * Math.ceil(width / 4);

			var val = this._mant.shl(padded_width - width);
			ret += "0x1.";

			for (var i = padded_width - 4; i >= 0; i -= 4) {
				var byte = val.shr(i).double_value();
				ret += byte.toString(16);

				val = val.and(BigNum.from_int32(15).shl(i).not(padded_width));

				if (val.is_0())
					break;
			}

			ret += "p" + (exp >= 0 ? "+" : "") + exp.toString(10);
		}
	}

	return ret;
}

Float.from_percent_a_string = function (str, config) {
	var sign = 0;
	var exp = 0;
	var mant = 0;

	str = str.trim();

	if (str.length < 3 || str.length > 30)
		return undefined;

	if (str.charAt(0) == "-") {
		sign = 1;
		str = str.substring(1);
	}
	else if (str.charAt(0) == "+") {
		str = str.substring(1);
	}

	str = str.toLowerCase();

	if (str === "inf") {
		return new Float(sign ? config.minus_inf_value : config.plus_inf_value, config);
	}

	if (str === "nan")
		return new Float(config.nan_value, config);

	if (str.charAt(0) != "0" || str.charAt(1) != "x")
		return undefined;

	str = str.substring(2);

	if (str.charAt(0) !== "1") {
		if (str === "0p+0")
			return Float.from_components(sign, exp, mant, config);

		return undefined;
	}

	str = str.substring(1);


	if (str.charAt(0) === "p") {
		var pexp = BigNum.from_string(str.substring(1), 10);

		if (pexp.cmp(config.max_biased_exp) > 0 || pexp.cmp(config.min_biased_exp.sub(config.mant_width - 1)) < 0)
			return undefined;

		if (pexp.cmp(config.min_biased_exp) <= 0) {
			var shift = config.min_biased_exp.sub(pexp);
			mant = BigNum.one.shl(config.mant_width - 1).shr(shift.double_value);
		} else {
			exp = pexp.add(config.max_biased_exp);
		}

		return Float.from_components(sign, exp, mant, config);
	}

	if (str.charAt(0) !== ".")
		return undefined;

	str = str.substring(1);

	var val = BigNum.zero();
	var shift = 0;
	while (str.length > 0 && str.charAt(0) != "p") {

		var digit = parseInt(str.charAt(0), 16);
		if (isNaN(digit))
			return undefined;

		val = val.shl(4);
		val = val.add(digit);
		str = str.substring(1);
		shift++;
	}


	var max_shift = Math.ceil(config.mant_width / 4);

	if (shift > max_shift) {
		return undefined;
	}

	val = val.shl(config.mant_width - shift * 4);

	if (str.charAt(0) == 'p') {
		var pexp = BigNum.from_string(str.substring(1), 10);


		if (pexp.cmp(config.max_biased_exp) > 0 || pexp.cmp(config.min_biased_exp.sub(config.mant_width - 1)) < 0)
			return undefined;

		if (pexp.cmp(config.min_biased_exp) <= 0) {
			val = val.or(BigNum.one().shl(config.mant_width));

			var slide = config.min_biased_exp.add(BigNum.one()).sub(pexp);
			val = val.shr(slide.double_value());
			mant = val;

			return Float.from_components(sign, exp, mant, config);

		}
		else {
			exp = pexp.add(config.max_biased_exp);
			mant = val;

			return Float.from_components(sign, exp, mant, config);

		}

	}

	return undefined;
}

Float.prototype.is_plus_inf = function () {
	return this._exp.eq(this._config.max_exp) && this._mant.is_0() && this._sign.is_0();
}

Float.prototype.is_minus_inf = function () {
	return this._exp.eq(this._config.max_exp) && this._mant.is_0() && this._sign.is_1();
}

Float.prototype.is_nan = function () {
	return this._exp.eq(this._config.max_exp) && !this._mant.is_0();
}

Float.prototype.log2 = function () {
	if (this.is_minus_inf() || this.is_plus_inf() || this.is_nan())
		return undefined;

	/* log(ab) = log(a) + log(b) */

	var val = this.biased_exp().double_value();

	var mant = this.mant.or(this.implicit_bit().shl(this._config.mant_width));

	/* We may lose one bit of precision with doubles here, but I don't think it matters in practice. */
	val += Math.log2(mant.double_value()) - this._config.mant_width;

	return val;
}


function trim_trailing_zeros(str) {
	for (var i = str.length - 1; i >= 0; i--) {
		if (str.charAt(i) !== "0")
			break;
	}

	return str.substring(0, Math.max(i + 1, 1));
}
