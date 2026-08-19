package osaka

import (
	"context"
	"fmt"

	"github.com/cockroachdb/errors"
	"github.com/vegidio/open-photo-ai/internal"
	"github.com/vegidio/open-photo-ai/internal/deps"
	"github.com/vegidio/open-photo-ai/internal/utils"
	"github.com/vegidio/open-photo-ai/types"
)

// The three graphs behind one Osaka model. Unlike the convolutional upscalers, which hold one session per scale
// factor, these are three stages of a single pass and are always loaded together.
const (
	ditSuffix = ""
	encSuffix = "_vae_encoder"
	decSuffix = "_vae_decoder"
)

// sessionSpec names one graph and the tensors it takes and returns. The names are not shared with the rest of the
// codebase, which uses "input"/"output" everywhere: these graphs were exported with meaningful names.
type sessionSpec struct {
	suffix  string
	inputs  []string
	outputs []string
}

var sessionSpecs = []sessionSpec{
	{suffix: ditSuffix, inputs: []string{"vid_input", "timestep"}, outputs: []string{"denoised_latent"}},
	{suffix: encSuffix, inputs: []string{"pixel_image"}, outputs: []string{"latent"}},
	{suffix: decSuffix, inputs: []string{"latent"}, outputs: []string{"pixel_image"}},
}

// profileFor is the provider tuning every Osaka graph needs.
//
// All three graphs have dynamic spatial axes, so ONNX Runtime's memory-pattern planner must be off: it assumes shapes
// repeat, and otherwise reserves for the largest region seen and never releases it. DynamicShapes says the same thing
// to any provider that would otherwise be told to expect fixed inputs.
//
// CoreML is supported. It matches the CPU result (no non-finite values, worst element difference 0.58 on a +-6.8
// range, which is fp16 accumulation order) at roughly CPU speed. Compiling the 6.8 GB graph takes about five minutes,
// but that happens once and is then served from the CoreML cache, so it is a first-run cost rather than a per-run one.
//
// TensorRT is the one exclusion. It needs explicit optimization profiles for dynamic inputs, and without them it
// either rebuilds an engine for every distinct tile size - minutes each - or grows an unbounded engine cache. Adding
// them means committing to shape ranges this pipeline does not yet have measurements for.
func profileFor() utils.EPProfile {
	return utils.EPProfile{
		DynamicShapes:     true,
		DisableMemPattern: true,
		DisableOptimizers: brokenOptimizers,
		ExcludeEPs:        []types.ExecutionProvider{types.ExecutionProviderTensorRT},
	}
}

func modelId(suffix string, precision types.Precision) string {
	return fmt.Sprintf("up_osaka%s_%s", suffix, precision)
}

// loadSessions downloads and opens the three graphs, in the order sessionSpecs lists them.
func loadSessions(
	ctx context.Context,
	precision types.Precision,
	ep types.ExecutionProvider,
	onProgress types.DownloadProgress,
) (_ utils.Sessions, retErr error) {
	sessions := make(utils.Sessions, 0, len(sessionSpecs))

	// Release whatever opened successfully if a later graph fails; the DiT alone is nearly 7 GB, so leaking it while
	// returning an error would leave the process holding memory nothing can reach.
	defer func() {
		if retErr != nil {
			sessions.Destroy()
		}
	}()

	profile := profileFor()

	for _, spec := range sessionSpecs {
		id := modelId(spec.suffix, precision)

		if err := deps.Install(ctx, deps.ModelDependency(id), onProgress); err != nil {
			return nil, errors.Wrapf(err, "failed to prepare the %s model", id)
		}

		internal.Log().Debug("loading model session", "model_id", id)

		session, err := utils.CreateSession(id+".onnx", spec.inputs, spec.outputs, ep, profile)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to create the %s session", id)
		}

		internal.Log().Debug("model session ready", "model_id", id)
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// brokenOptimizers are the ONNX Runtime graph transformers that miscompile this DiT.
//
// They rewrite the graph so it refers to a tensor they removed, and session creation then fails with a "name which
// does not exist" error naming a node they created - most recently
// "InsertedPrecisionFreeCast_/dit/vid_out_norm/Constant_output_0" for a SimplifiedLayerNormFusion node.
//
// This is a runtime bug, not an export one, and it is version-specific: the graph loads cleanly under ONNX Runtime
// 1.29, and fails under the 1.26 this app bundles. So it cannot be retired by re-exporting, only by moving the
// bundled runtime forward - and it must be checked against the bundled build rather than whatever a local Python
// install happens to have, which is how it was briefly and wrongly declared fixed.
var brokenOptimizers = []string{"ReshapeFusion", "SimplifiedLayerNormFusion"}
