<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;
use Rindow\NeuralNetworks\Gradient\Variable;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Core\GradientUtils;
use Rindow\NeuralNetworks\Gradient\MaskedNDArray;


/**
 *
 */
abstract class AbstractAttentionLayer extends AbstractLayerBase
{
    use GenericUtils;
    use GradientUtils;

    /** @var array<int>|null $scoresShape */
    protected ?array $scoresShape=null;

    /**
     * @param array<NDArray> $inputs
     * @param array<NDArray|null> $mask
     * @return array<Variable>|Variable
     */
    abstract protected function call(
        array $inputs,
        ?bool $training=null,
        ?bool $returnAttentionScores=null,
        ?array $mask=null,
    ) : NDArray|array;

    /**
     * @return array<NDArray>
     */
    abstract protected function differentiate(NDArray $dOutputs) : array;

    /**
     * @param array<NDArray> $inputs
     */
    protected function assertInputShapes(array $inputs,string $direction) : void
    {
        if(!$this->shapeInspection)
            return;
        if($this->inputShape===null) {
            throw new InvalidArgumentException('Uninitialized input shape in '.$this->name.':'.$direction);
        }
        if(count($inputs)!=2 && count($inputs)!=3) {
            throw new InvalidArgumentException('Must have 2 or 3 arguments in '.$this->name.':'.$direction);
        }
        //$tq = $this->inputShape[0][0];
        //$dim = $this->inputShape[0][1];
        //$tv = $this->inputShape[1][0];
        $qshape = $inputs[0]->shape();
        $batchNum = array_shift($qshape);
        $vshape = $inputs[1]->shape();
        $vbatchNum = array_shift($vshape);
        if($batchNum!=$vbatchNum) {
            throw new InvalidArgumentException('Unmatch batch size of query and value: '.
                "query=[$batchNum,".implode(',',$qshape)."],".
                "value=[$vbatchNum,".implode(',',$vshape)."]".
                "in ".$this->name.':'.$direction);
        }
        if($this->inputShape[0]!=$qshape){
            throw new InvalidArgumentException('Unmatch query shape '.
                ' [b,'.implode(',',$this->inputShape[0]).'] NDArray.'.
                ' ['.$batchNum.','.implode(',',$qshape).'] given in '.$this->name.':'.$direction);
        }
        if($this->inputShape[1]!=$vshape){
            throw new InvalidArgumentException('Unmatch value shape '.
                ' [b,'.implode(',',$this->inputShape[1]).'] NDArray.'.
                ' ['.$vbatchNum.','.implode(',',$vshape).'] given in '.$this->name.':'.$direction);
        }
        if(count($inputs)==3) {
            $kshape = $inputs[2]->shape();
            $kbatchNum = array_shift($kshape);
            if($vbatchNum!=$kbatchNum) {
                throw new InvalidArgumentException('Unmatch batch size of value and key: '.
                    "query=[$vbatchNum,".implode(',',$vshape)."],".
                    "value=[$kbatchNum,".implode(',',$kshape)."]".
                    "in ".$this->name.':'.$direction);
            }
            if($kshape!=$vshape){
                throw new InvalidArgumentException('Unmatch value shape and key shape.:'.
                    ' ['.implode(',',$vshape).'],['.implode(',',$kshape).'] in '.$this->name.':'.$direction);
            }
        }
    }

    protected function assertScoresShape(NDArray $scores,string $direction) : void
    {
        if(!$this->shapeInspection) {
            return;
        }
        if($this->scoresShape===null) {
            throw new InvalidArgumentException('Uninitialized scores shape');
        }
        $shape = $scores->shape();
        $batchNum = array_shift($shape);
        if($shape!=$this->scoresShape) {
            $shape = $this->shapeToString($shape);
            $scoresShape = $this->shapeToString($this->scoresShape);
            throw new InvalidArgumentException('unmatch scores shape: '.$shape.', must be '.$scoresShape.' in '.$this->name.':'.$direction);
        }
    }

    /**
     * @param array<NDArray> $dOutputs
     * @param ArrayAccess<object,object> $grads
     * @param array<NDArray> $oidsToCollect
     * @return array<NDArray>
     */
    public function backward(
        array $dOutputs,
        ?ArrayAccess $grads=null,
        ?array $oidsToCollect=null
        ) : array
    {
        if(count($dOutputs)!=1&&count($dOutputs)!=2) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $dOutputs = $dOutputs[0];
        if(!($dOutputs instanceof NDArray)) {
            throw new InvalidArgumentException('dOutputs must be list containing one NDArray');
        }
        $this->assertOutputShape($dOutputs,'backward');
        $dInputs = $this->differentiate($dOutputs);
        $this->assertInputShapes($dInputs,'backward');
        $this->collectGradients($this->backend,array_map(null,$this->trainableVariables(),$this->getGrads()),
            $grads,$oidsToCollect);
        return $dInputs;
    }

    /**
     * @param array<string,mixed> $options
     */
    protected function numOfOutputs(?array $options) : int
    {
        if($options['returnAttentionScores'] ?? false) {
            return 2;
        }
        return 1;
    }

    /**
     * @return array<Variable>|Variable
     */
    final public function __invoke(mixed ...$args) : array|NDArray
    {
        return $this->forward(...$args);
    }

    /**
     * Call from SessionFunc in compiled graph
     * @param array<NDArray> $inputs
     * @param array<string,mixed> $options
     * @return array<NDArray>
     */
    public function _rawCall(array $inputs,array $options) : array
    {
        $queryMask = $options['queryMask'] ?? null;
        $valueMask = $options['valueMask'] ?? null;
        $keyMask   = $options['keyMask'] ?? null;
        $mask = null;
        if($queryMask!==null || $valueMask!==null || $keyMask!==null) {
            $mask = [$queryMask,$valueMask];
            if($keyMask!==null) {
                $mask[] = $keyMask;
            }
        } else {
            $mask = $this->retrieveMultiMasks($inputs);
        }
        unset($options['queryMask']);
        unset($options['valueMask']);
        unset($options['keyMask']);
        $options['mask'] = $mask;
        $outputs = $this->call(
            $inputs,
            ...$options,
        );
        if(!is_array($outputs)) {
            $outputs = [$outputs];
        }
        $outputs[0] = $this->makeSingleMaskedValue($inputs[0], $outputs[0]);
        return $outputs;
    }

    public function computeMask(
        array|NDArray $inputs,
        array|NDArray|null $previousMask
        ) : array|NDArray|null
    {
        return $previousMask;
    }
}
