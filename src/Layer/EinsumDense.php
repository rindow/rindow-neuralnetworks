<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use LogicException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class EinsumDense extends AbstractLayer
{
    use GenericUtils;
    protected int $units;
    protected bool $useBias;
    protected mixed $kernelInitializer;
    protected mixed $biasInitializer;
    protected ?string $kernelInitializerName;
    protected ?string $biasInitializerName;

    protected ?NDArray $kernel=null;
    protected NDArray $bias;
    protected NDArray $dKernel;
    protected NDArray $dBias;

    protected string $equation;
    /** @var array<int> $partial_output_shape */
    protected array $partial_output_shape;
    /** @var array<int> $full_output_shape */
    protected array $full_output_shape;
    protected ?string $bias_axes;
    protected string $dInputsBackwardEquation;
    protected string $dKernelBackwardEquation;

    /**
     * @param int|array<int> $output_shape
     * @param array<int>|null $input_shape
     */
    public function __construct(
        object $backend,
        string $equation,
        int|array $output_shape,
        ?array $input_shape=null,
        string|object|null $activation=null,
        ?string $bias_axes=null,
        string|object|null $kernel_initializer=null,
        string|object|null $bias_initializer=null,
        ?string $name=null,
    )
    {
        $kernel_initializer ??= 'glorot_uniform';
        $bias_initializer ??= 'zeros';

        $this->equation = $equation;
        if(is_int($output_shape)) {
            $output_shape = [$output_shape];
        }
        parent::__construct($backend);
        $K = $backend;
        $this->inputShape = $input_shape;
        $this->partial_output_shape = $output_shape;
        $this->bias_axes = $bias_axes;
        $this->useBias = ($this->bias_axes!==null) ? true : false;
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $this->toStringName($kernel_initializer);
        $this->biasInitializerName = $this->toStringName($bias_initializer);
        $this->initName($name,'einsumdense');
        $this->allocateWeights($this->useBias?['kernel','bias']:['kernel']);
        $this->setActivation($activation);
    }

    public function getEquation() : string
    {
        return $this->equation;
    }
    
    public function build(mixed $variable=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($variable);

        [
            $kernel_shape,
            $bias_shape,
            $full_output_shape,
            $backward_dinput_equation,
            $backward_dkernel_equation
        ] = $this->analyze_einsum_string(
            $this->equation,
            $this->bias_axes,
            $inputShape,
            $this->partial_output_shape,
        );
        $this->dInputsBackwardEquation  = $backward_dinput_equation;
        $this->dKernelBackwardEquation  = $backward_dkernel_equation;
        $this->full_output_shape = $full_output_shape;

        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
                if($this->useBias) {
                    $this->bias = $sampleWeights[1];
                }
            } else {
                $fan_in = (int)array_product($this->inputShape);
                $fan_out = (int)array_product($this->partial_output_shape);
                $this->kernel = $kernelInitializer(
                    $kernel_shape,
                    [$fan_in,$fan_out]);
                if($this->useBias) {
                    $this->bias = $biasInitializer(
                        $bias_shape);
                }
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        $output_shape = $full_output_shape;
        array_shift($output_shape);
        $this->outputShape = $output_shape;
        $this->syncWeightVariables();
        $this->built = true;

    }

    public function getParams() : array
    {
        if($this->useBias) {
            return [$this->kernel,$this->bias];
        } else {
            return [$this->kernel];
        }
    }

    public function getGrads() : array
    {
        if($this->useBias) {
            return [$this->dKernel,$this->dBias];
        } else {
            return [$this->dKernel];
        }
    }

    public function reverseSyncWeightVariables() : void
    {
        if($this->useBias) {
            $this->kernel = $this->weights[0]->value();
            $this->bias = $this->weights[1]->value();
        } else {
            $this->kernel = $this->weights[0]->value();
        }
    }

    public function kernel() : NDArray
    {
        if(!$this->built) {
            throw new LogicException(
                "You must build the layer before accessing `kernel`."
            );
        }
        return $this->kernel;
    }

    /**
     * @return array<int>
     */
    public function compute_output_shape() : array
    {
        return $this->full_output_shape;
    }
    
    protected function call(NDArray $inputs, ?bool $training=null) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        $container->inputs = $inputs;
        $outputs = $K->einsum($this->equation, $inputs, $this->kernel());
        if($this->useBias) {
            $K->update_add($outputs,$this->bias);
        }
        if($this->activation) {
            $container->activation = new \stdClass();
            $outputs = $this->activation->forward($container->activation,$outputs,training:$training);
        }
        return $outputs;
    }
    
    protected function differentiate(NDArray $dOutputs) : NDArray
    {
        $K = $this->backend;
        $container = $this->container();
        if($this->activation) {
            $dOutputs = $this->activation->backward($container->activation,$dOutputs);
        }
        $dInputs = $K->einsum($this->dInputsBackwardEquation, $dOutputs, $this->kernel());

        // update params
        $dKernel = $K->einsum($this->dKernelBackwardEquation, $dOutputs, $container->inputs);
        $K->copy($dKernel,$this->dKernel);
        if($this->useBias) {
            $biasFlatSize = (int)array_product($this->dBias->shape());
            $dOutputsFlatSize = (int)array_product($dOutputs->shape());
            $dOutputsFlat = $dOutputs->reshape([intdiv($dOutputsFlatSize,$biasFlatSize),$biasFlatSize]);
            $dBiasFlat = $this->dBias->reshape([$biasFlatSize]);
            $K->sum($dOutputsFlat, axis:0, output:$dBiasFlat);
        }

        return $dInputs;
    }
    
    /**
     * @return array<string,mixed>
     */
    public function get_config() : array
    {
        $config = [
            "output_shape"=>$this->partial_output_shape,
            "equation"=>$this->equation,
            "activation"=>$this->activationName,
            "bias_axes"=>$this->bias_axes,
            "kernel_initializer"=>$this->kernelInitializerName,
            "bias_initializer"=>$this->biasInitializerName,
        ];
        return $config;
    }
    
    /**
     * Analyzes an einsum string to determine the required weight shape.
     * 
     * return [
     *     $kernel_shape,
     *     $bias_shape,
     *     $full_output_shape,
     *     $backward_dinput_equation,
     *     $backward_dkernel_equation
     * ]
     * 
     *  @param array<int> $input_shape
     *  @param array<int> $output_shape
     *  @return array{array<int>,array<int>,array<int>,string,string}
     */
    private function analyze_einsum_string(
        string $equation,
        ?string $bias_axes,
        array $input_shape,
        array $output_shape
    ) : array
    {
        //echo "equation=$equation\n";
        //echo "einsum analyze_einsum_string output_shape arg=(".implode(',',$output_shape).")\n";
        //$dot_replaced_string = $re->sub("\.\.\.", "0", $equation);
        $dot_replaced_string = str_replace("...", "0", $equation);
        # This is the case where no ellipses are present in the string.
        //$split_string = $re->match(
        //    "([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)", $dot_replaced_string
        //);
        preg_match(
            "/([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)/",
            $dot_replaced_string,
            $split_string
        );
        if($bias_axes!==null) {
            $bias_axes = str_split($bias_axes);
        }
        if($split_string) {
            [$dmy,$input_chrs,$weight_chrs,$output_chrs] = $split_string;
            if(count($input_shape)+1!=strlen($input_chrs)) {
                throw new InvalidArgumentException('Unmatch rank of input_shape and input spec in equation');
            }
            if(count($output_shape)+1!=strlen($output_chrs)) {
                throw new InvalidArgumentException('Unmatch rank of output_shape and output spec in equation');
            }
            $shapes = $this->analyze_split_string(
                $split_string, $bias_axes, $input_shape, $output_shape
            );
            $backward_dinput_equation  = "{$output_chrs},{$weight_chrs}->{$input_chrs}";
            $backward_dkernel_equation = "{$output_chrs},{$input_chrs}->{$weight_chrs}";
            return array_merge($shapes,[$backward_dinput_equation,$backward_dkernel_equation]);
        }
    
        # This is the case where ellipses are present on the left.
        //$split_string = $re->match(
        //    "0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)", $dot_replaced_string
        //);
        preg_match(
            "/0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)/",
            $dot_replaced_string,
            $split_string
        );
        if($split_string) {
            [$dmy,$input_chrs,$weight_chrs,$output_chrs] = $split_string;
            $shapes = $this->analyze_split_string(
                $split_string, $bias_axes, $input_shape, $output_shape, $left_elided=true
            );
            $backward_dinput_equation  = "...{$output_chrs},{$weight_chrs}->...{$input_chrs}";
            $backward_dkernel_equation = "...{$output_chrs},...{$input_chrs}->{$weight_chrs}";
            return array_merge($shapes,[$backward_dinput_equation,$backward_dkernel_equation]);
        }
    
        # This is the case where ellipses are present on the right.
        //$split_string = $re->match(
        //    "([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0", $dot_replaced_string
        //);
        preg_match(
            "/([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0/",
            $dot_replaced_string,
            $split_string
        );
        if($split_string) {
            [$dmy,$input_chrs,$weight_chrs,$output_chrs] = $split_string;
            $shapes = $this->analyze_split_string(
                $split_string, $bias_axes, $input_shape, $output_shape
            );
            $backward_dinput_equation  = "{$output_chrs}...,{$weight_chrs}->{$input_chrs}...";
            $backward_dkernel_equation = "{$output_chrs}...,{$input_chrs}...->{$weight_chrs}";
            return array_merge($shapes,[$backward_dinput_equation,$backward_dkernel_equation]);
        }
    
        throw new InvalidArgumentException(
            "Invalid einsum equation '{$equation}'. Equations must be in the form ".
            "[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]...."
        );
    }
    

    /**
     *  Analyze an pre-split einsum string to find the weight shape.
     * 
     *  @param array{string,string,string,string} $split_string
     *  @param array<string>|null $bias_axes
     *  @param array<int>|null $input_shape 
     *  @param array<int> $output_shape
     *  @return array{array<int>,array<int>,array<int>}
     */
    private function analyze_split_string(
        array $split_string,
        ?array $bias_axes,
        ?array $input_shape,
        array $output_shape,
        ?bool $left_elided=null
    ) : array
    {
        //echo "einsum analyze_split_string input_shape arg=(".implode(',',$input_shape).")\n";
        //echo "einsum analyze_split_string output_shape arg=(".implode(',',$output_shape).")\n";
        $left_elided ??= false;
        $input_spec = str_split($split_string[1]);
        $weight_spec = str_split($split_string[2]);
        $output_spec = str_split($split_string[3]);

        array_unshift($input_shape, 1);  // add batch shape
        //echo "input_shape rank=".count($input_shape)."\n";
        //echo "input_spec rank=".count($input_spec)."\n";
        $elided = count($input_shape) - count($input_spec);

        array_unshift($output_shape, $input_shape[0]);
        //echo "einsum analyze_split_string output_shape array_unshift=(".implode(',',$output_shape).")\n";
        //echo "elided=";var_dump($elided);
        //echo "left_elided=";var_dump($left_elided);
        if($elided > 0 && $left_elided) {
            $top = array_shift($output_shape);
            for($i=1; $i<$elided; $i++) {
                # We already inserted the 0th input dimension at dim 0, so we need
                # to start at location 1 here.
                array_unshift($output_shape,$input_shape[$i]);
            }
            array_unshift($output_shape,$top);
        } elseif($elided > 0 && !$left_elided) {
            $count = count($input_shape);
            for($i=count($input_shape) - $elided; $i<$count; $i++) {
                array_push($output_shape,$input_shape[$i]);
            }
        }
        //echo "einsum analyze_split_string output_shape format=(".implode(',',$output_shape).")\n";

        if($left_elided) {
            # If we have beginning dimensions elided, we need to use negative
            # indexing to determine where in the input dimension our values are.
            $input_dim_map = [];
            foreach($input_spec as $i=>$dim) {
                $pos = ($i + $elided) - count($input_shape);
                $pos = ($pos<0) ? count($input_shape)+$pos : $pos;
                $input_dim_map[$dim] = $pos;
            }
            # Because we've constructed the full output shape already, we don't need
            # to do negative indexing.
            $output_dim_map = [];
            foreach($output_spec as $i=>$dim) {
                $output_dim_map[$dim] = $i + $elided;
            }
        } else {
            $input_dim_map = array_flip($input_spec);
            $output_dim_map = array_flip($output_spec);
        }
    
        foreach($input_spec as $dim) {
            $input_shape_at_dim = $input_shape[$input_dim_map[$dim]];
            if(in_array($dim,$output_dim_map)) {
                $output_shape_at_dim = $output_shape[$output_dim_map[$dim]];
                if(
                    $output_shape_at_dim !==null &&             // NOT free dim
                    $output_shape_at_dim != $input_shape_at_dim // fixed dim
                ) {
                    throw new InvalidArgumentException(
                        "Input shape and output shape do not match at shared ".
                        "dimension '{$dim}'. Input shape is {$input_shape_at_dim}, ".
                        "and output shape ".
                        "is ".$output_shape[$output_dim_map[$dim]]."."
                    );
                }
            }
        }
    
        foreach($output_spec as $dim) {
            if(!in_array($dim,$input_spec) && !in_array($dim,$weight_spec)) {
                throw new InvalidArgumentException(
                    "Dimension '{$dim}' was specified in the output ".
                    "'".implode(',',$output_spec)."' but has no corresponding dim in the input ".
                    "spec '".implode(',',$input_spec)."' or weight spec '".implode(',',$output_spec)."'"
                );
            }
        }
    
        $weight_shape = [];
        foreach($weight_spec as $dim) {
            if(array_key_exists($dim,$input_dim_map)) {
                array_push($weight_shape,$input_shape[$input_dim_map[$dim]]);
            } elseif(array_key_exists($dim,$output_dim_map)) {
                array_push($weight_shape,$output_shape[$output_dim_map[$dim]]);
            } else {
                throw new InvalidArgumentException(
                    "Weight dimension '{$dim}' did not have a match in either ".
                    "the input spec '".implode(',',$input_spec)."' or the output ".
                    "spec '".implode(',',$output_spec)."'. For this layer, the weight must ".
                    "be fully specified."
                );
            }
        }

        if($bias_axes) {
            $num_left_elided = ($left_elided) ? $elided : 0;
            $idx_map = [];
            foreach($output_spec as $i=>$char) {
                $idx_map[$char] = $output_shape[$i + $num_left_elided];
            }
    
            foreach($bias_axes as $char) {
                if(!in_array($char,$output_spec)) {
                    throw new InvalidArgumentException(
                        "Bias dimension '{$char}' was requested, but is not part ".
                        "of the output spec '".implode(',',$output_spec)."'"
                    );
                }
            }
            $flip_output_spec = array_flip($output_spec);
            $first_bias_location = min(
                array_map(fn($char)=>$flip_output_spec[$char],$bias_axes)
            );
            $bias_output_spec = array_slice($output_spec,$first_bias_location);
    
            $bias_shape = array_map(
                    fn($char)=>(in_array($char,$bias_axes))?$idx_map[$char]:1, 
                    $bias_output_spec
            );
    
            if(!$left_elided) {
                for($i=0;$i<$elided;++$i) {
                    $bias_shape[] = 1;
                }
            }
        } else {
            $bias_shape = null;
        }
    
        return [$weight_shape, $bias_shape, $output_shape];
    }
    
}
