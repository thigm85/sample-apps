// Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

package ai.vespa.processor;

import ai.vespa.tokenizer.WordPieceTokenizer;
import com.google.inject.Inject;
import com.yahoo.docproc.DocumentProcessor;
import com.yahoo.docproc.Processing;
import com.yahoo.document.Document;
import com.yahoo.document.DocumentOperation;
import com.yahoo.document.DocumentPut;
import com.yahoo.document.datatypes.FieldValue;
import com.yahoo.document.datatypes.StringFieldValue;
import com.yahoo.document.datatypes.TensorFieldValue;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;


public class TransformerProcessor extends DocumentProcessor {

    WordPieceTokenizer tokenizer;

    private static final TensorType TOKEN_TENSOR_TYPE = TensorType.fromSpec("tensor<float>(d0[128])");

    @Inject
    public TransformerProcessor(WordPieceTokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public Progress process(Processing processing) {
        for (DocumentOperation op : processing.getDocumentOperations()) {
            if (op instanceof DocumentPut) {
                DocumentPut put = (DocumentPut) op;
                Document doc = put.getDocument();
                if (!doc.getDataType().getName().equals("msmarco")) {
                    continue;
                }
                createTokenSequence(doc);
            }
        }
        return Progress.DONE;
    }

    private void createTokenSequence(Document doc) {
        int maxLength = TOKEN_TENSOR_TYPE.sizeOfDimension("d0").get().intValue();
        String title = getStringValue(doc, "title");
        String body = getStringValue(doc, "body");

        Tensor.Builder builder = Tensor.Builder.of(TOKEN_TENSOR_TYPE);
        int i = 0;
        for (Integer tokenId : tokenizer.tokenize(title + body, maxLength, true)) {
            builder.cell(tokenId, i++);
        }
        doc.setFieldValue("tokens", new TensorFieldValue(builder.build()));
    }

    private String getStringValue(Document doc, String fieldName) {
        FieldValue fieldValue = doc.getFieldValue(fieldName);
        if (!(fieldValue instanceof StringFieldValue)) {
            throw new IllegalArgumentException("Can only create tensor from string field input");
        }
        return ((StringFieldValue) fieldValue).getString();
    }

}
