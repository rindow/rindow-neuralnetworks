<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Support\HDA\HDASqliteFactory;

class Utils
{
    protected object $backend;

    public function __construct(object $backend)
    {
        $this->backend = $backend;
    }

    public function HDA(mixed $type=null) : object
    {
        return new HDASqliteFactory();
    }
}
