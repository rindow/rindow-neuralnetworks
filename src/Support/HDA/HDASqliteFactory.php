<?php
namespace Rindow\NeuralNetworks\Support\HDA;

class HDASqliteFactory
{
    public function open(string $filename, string $mode=null)
    {
        return new HDASqlite($filename, $mode);
    }
}
