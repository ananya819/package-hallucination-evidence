package com.example.tensorflowext;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import com.tensorflow.ext.CustomOp;
import com.tensorflow.ext.OpContext;

public class CustomReluOp extends CustomOp {

    @Override
    public String getName() {
        return "CustomRelu";
    }

    @Override
    public Tensor<?> compute(OpContext context, Tensor<?> input) {
        float[] data = new float[(int) input.shape().size(0)];
        input.copyTo(data);

        for (int i = 0; i < data.length; i++) {
            data[i] = Math.max(0, data[i]); // ReLU activation
        }

        return Tensor.of(Shape.of(data.length), d -> {
            for (int i = 0; i < data.length; i++) {
                d.setFloat(data[i], i);
            }
        });
    }
}
